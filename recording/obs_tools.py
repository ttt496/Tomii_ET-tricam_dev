"""Helpers for integrating OBS WebSocket camera metadata with OpenCV indices."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import platform
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import cv2
import websockets
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from recording.capture import list_available_cameras
else:
    from .capture import list_available_cameras


@dataclass(frozen=True)
class OBSInputInfo:
    """Camera-like input registered inside OBS."""

    name: str
    kind: str
    enabled: bool
    settings: Dict[str, Any]
    uuid: Optional[str]
    scene_item_id: Optional[int]
    raw: Dict[str, Any]

    def device_descriptor(self) -> Optional[str]:
        """Best-effort unique device description sourced from settings."""
        candidates: Iterable[str | None] = (
            self.settings.get("device_path"),
            self.settings.get("devicePath"),
            self.settings.get("device_id"),
            self.settings.get("deviceId"),
            self.settings.get("video_device_id"),
            self.settings.get("videoDeviceId"),
            self.settings.get("device"),
            self.settings.get("deviceName"),
            self.settings.get("video_device"),
            self.settings.get("videoDevice"),
            self.settings.get("input"),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return None

    def guess_index(self) -> Optional[int]:
        descriptor = self.device_descriptor()
        if not descriptor:
            return None

        patterns = [
            r"/dev/video(?P<idx>\d+)",
            r"video(?P<idx>\d+)",
            r"camera_(?P<idx>\d+)",
            r"#(?P<idx>\d+)$",
            r"\\\\?\\?usb.*#(?P<idx>\d+)$",
            r"device(?P<idx>\d+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, descriptor, re.IGNORECASE)
            if match:
                try:
                    return int(match.group("idx"))
                except (ValueError, KeyError):
                    continue
        if descriptor.isdigit():
            return int(descriptor)
        return None


@dataclass(frozen=True)
class CameraIndexInfo:
    index: int
    width: Optional[float]
    height: Optional[float]
    fps: Optional[float]
    backend: Optional[str]


@dataclass(frozen=True)
class CameraMapping:
    obs_input: OBSInputInfo
    guessed_index: Optional[int]
    matched_index: Optional[int]
    camera_info: Optional[CameraIndexInfo]
    notes: Optional[str] = None


class OBSWebSocketError(RuntimeError):
    """Raised when OBS WebSocket communication fails."""


async def _recv_json(ws: websockets.WebSocketClientProtocol, *, timeout: float) -> Dict[str, Any]:
    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(message)


def _compute_auth_response(password: str, challenge: str, salt: str) -> str:
    secret = base64.b64encode(hashlib.sha256((password + salt).encode("utf-8")).digest()).decode("utf-8")
    auth = base64.b64encode(hashlib.sha256((secret + challenge).encode("utf-8")).digest()).decode("utf-8")
    return auth


def _build_ws_uri(host: str, port: int) -> str:
    host = host.strip()
    if host.startswith("ws://") or host.startswith("wss://"):
        return host
    # Wrap raw IPv6 literals with brackets for RFC 3986 compliance.
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"ws://{host}:{port}"


async def _identify(
    host: str,
    port: int,
    *,
    password: Optional[str],
    timeout: float,
) -> websockets.WebSocketClientProtocol:
    uri = _build_ws_uri(host, port)
    ws = await websockets.connect(uri, open_timeout=timeout, close_timeout=timeout, ping_interval=None)

    hello = await _recv_json(ws, timeout=timeout)
    if hello.get("op") != 0:
        await ws.close()
        raise OBSWebSocketError(f"Expected Hello (op 0), received: {hello}")

    identify_payload: Dict[str, Any] = {
        "op": 1,
        "d": {
            "rpcVersion": hello["d"].get("rpcVersion", 1),
            "eventSubscriptions": 0,
        },
    }

    auth_data = hello["d"].get("authentication")
    if auth_data:
        if password is None:
            await ws.close()
            raise OBSWebSocketError("OBS WebSocket requires a password but none was provided.")
        identify_payload["d"]["authentication"] = _compute_auth_response(
            password, auth_data["challenge"], auth_data["salt"]
        )

    await ws.send(json.dumps(identify_payload))

    while True:
        message = await _recv_json(ws, timeout=timeout)
        op = message.get("op")
        if op == 2:
            return ws
        if op in {5, 7}:  # events or stray responses before identification
            continue
        await ws.close()
        raise OBSWebSocketError(f"Unexpected message during identification: {message}")


async def _send_request(
    ws: websockets.WebSocketClientProtocol,
    request_type: str,
    request_data: Optional[Dict[str, Any]] = None,
    *,
    timeout: float,
) -> Dict[str, Any]:
    request_id = uuid.uuid4().hex
    payload = {
        "op": 6,
        "d": {
            "requestType": request_type,
            "requestId": request_id,
        },
    }
    if request_data:
        payload["d"]["requestData"] = request_data

    await ws.send(json.dumps(payload))

    while True:
        message = await _recv_json(ws, timeout=timeout)
        op = message.get("op")
        if op == 5:
            continue  # Ignore events
        if op != 7:
            raise OBSWebSocketError(f"Unexpected message type: {message}")

        data = message["d"]
        if data.get("requestId") != request_id:
            continue

        status = data.get("requestStatus", {})
        if not status.get("result"):
            code = status.get("code")
            comment = status.get("comment")
            raise OBSWebSocketError(f"OBS request {request_type} failed (code={code}): {comment}")
        return data


async def _fetch_obs_inputs_async(
    host: str,
    port: int,
    *,
    password: Optional[str],
    timeout: float,
    kinds: Optional[Sequence[str]] = None,
) -> List[OBSInputInfo]:
    ws = await _identify(host, port, password=password, timeout=timeout)
    try:
        response = await _send_request(ws, "GetInputList", timeout=timeout)
        inputs = response.get("responseData", {}).get("inputs", [])

        desired_kinds = tuple(kinds) if kinds else None
        result: List[OBSInputInfo] = []
        for item in inputs:
            input_kind = item.get("inputKind", "")
            if desired_kinds is not None and input_kind not in desired_kinds:
                continue
            input_name = item.get("inputName", "")
            input_uuid = item.get("inputUuid")
            scene_item_id = item.get("sceneItemId")
            enabled = bool(item.get("inputEnabled", True))

            settings_response = await _send_request(
                ws,
                "GetInputSettings",
                {"inputName": input_name},
                timeout=timeout,
            )
            settings = settings_response.get("responseData", {}).get("inputSettings", {})
            result.append(
                OBSInputInfo(
                    name=input_name,
                    kind=input_kind,
                    enabled=enabled,
                    settings=settings,
                    uuid=input_uuid,
                    scene_item_id=scene_item_id,
                    raw=item,
                )
            )
        return result
    finally:
        await ws.close()


def fetch_obs_camera_inputs(
    host: str = "127.0.0.1",
    port: int = 4455,
    *,
    password: Optional[str] = None,
    timeout: float = 5.0,
    kinds: Optional[Sequence[str]] = None,
) -> List[OBSInputInfo]:
    """Return camera-like OBS inputs via WebSocket."""

    if kinds is None:
        desired_kinds: Optional[Sequence[str]] = (
            "dshow_input",
            "avcapture_input",
            "v4l2_input",
            "wasapi_input_capture",
            "ffmpeg_source",
        )
    elif not kinds:
        desired_kinds = None
    else:
        desired_kinds = kinds
    return asyncio.run(
        _fetch_obs_inputs_async(
            host,
            port,
            password=password,
            timeout=timeout,
            kinds=desired_kinds,
        )
    )


def probe_camera_indices(indices: Sequence[int]) -> List[CameraIndexInfo]:
    infos: List[CameraIndexInfo] = []
    for index in indices:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            infos.append(CameraIndexInfo(index=index, width=None, height=None, fps=None, backend=None))
            continue
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend_id = int(cap.get(cv2.CAP_PROP_BACKEND)) if hasattr(cv2, "CAP_PROP_BACKEND") else None
        backend_name = None
        if backend_id is not None:
            try:
                backend_name = cv2.videoio_registry.getBackendName(backend_id)
            except Exception:  # pylint: disable=broad-except
                backend_name = None
        cap.release()
        infos.append(CameraIndexInfo(index=index, width=width, height=height, fps=fps, backend=backend_name))
    return infos


def capture_snapshots(
    indices: Sequence[int],
    output_dir: Path,
    *,
    warmup_frames: int = 5,
    timeout_frames: int = 30,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for index in indices:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Warning: could not open camera {index} for snapshot.")
            continue
        frame: Optional[Any] = None
        for frame_idx in range(max(warmup_frames, 0) + max(timeout_frames, 1)):
            ret, current = cap.read()
            if not ret:
                frame = None
                continue
            frame = current
            if frame_idx >= warmup_frames:
                break
        cap.release()
        if frame is None:
            print(f"Warning: failed to capture frame for camera {index}.")
            continue
        file_path = output_dir / f"camera_{index}.jpg"
        if cv2.imwrite(str(file_path), frame):
            saved.append(file_path)
        else:
            print(f"Warning: failed to write snapshot for camera {index} at {file_path}.")
    return saved


def map_obs_inputs_to_indices(
    obs_inputs: Sequence[OBSInputInfo],
    *,
    max_devices: int = 10,
    preferred_indices: Optional[Sequence[int]] = None,
) -> List[CameraMapping]:
    """Best-effort match between OBS inputs and OpenCV device indices."""

    available_indices = list(preferred_indices) if preferred_indices else list_available_cameras(max_devices)
    camera_infos = {info.index: info for info in probe_camera_indices(available_indices)}

    mappings: List[CameraMapping] = []
    for obs_input in obs_inputs:
        guess = obs_input.guess_index()
        matched = guess if guess in camera_infos else None
        notes = None
        if guess is not None and matched is None:
            notes = f"Guessed index {guess} not in available device list"
        mappings.append(
            CameraMapping(
                obs_input=obs_input,
                guessed_index=guess,
                matched_index=matched,
                camera_info=camera_infos.get(matched) if matched is not None else None,
                notes=notes,
            )
        )
    return mappings


def _format_dimension(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "?"
    return f"{int(round(value))}"


def _print_mapping_table(mappings: Sequence[CameraMapping]) -> None:
    if not mappings:
        print("No OBS camera inputs detected.")
        return

    header = (
        "OBS Name",
        "Kind",
        "Descriptor",
        "Guessed Idx",
        "Matched Idx",
        "Resolution",
        "FPS",
        "Backend",
        "Notes",
    )
    print(" | ".join(header))
    print("-" * 110)

    for mapping in mappings:
        obs_input = mapping.obs_input
        descriptor = obs_input.device_descriptor() or "-"
        camera_info = mapping.camera_info
        resolution = (
            f"{_format_dimension(camera_info.width)}x{_format_dimension(camera_info.height)}"
            if camera_info
            else "-"
        )
        fps = f"{camera_info.fps:.2f}" if camera_info and camera_info.fps and camera_info.fps > 0 else "-"
        backend = camera_info.backend or "-" if camera_info else "-"
        row = (
            obs_input.name,
            obs_input.kind,
            descriptor,
            str(mapping.guessed_index) if mapping.guessed_index is not None else "-",
            str(mapping.matched_index) if mapping.matched_index is not None else "-",
            resolution,
            fps,
            backend,
            mapping.notes or "",
        )
        print(" | ".join(row))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Map OBS camera inputs to OpenCV device indices.")
    parser.add_argument("--host", default="localhost", help="OBS WebSocket host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4455, help="OBS WebSocket port (default: 4455)")
    parser.add_argument(
        "--password",
        help="OBS WebSocket password if required (overrides values loaded from the environment)",
    )
    parser.add_argument(
        "--password-env",
        default="OBS_WEBSOCKET_PASSWORD",
        help="Environment variable to read the password from when --password is omitted (default: OBS_WEBSOCKET_PASSWORD)",
    )
    parser.add_argument(
        "--dotenv",
        help="Optional path to a .env file containing OBS_WEBSOCKET_PASSWORD or similar secrets.",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout for WebSocket operations")
    parser.add_argument(
        "--max-devices",
        type=int,
        default=10,
        help="How many camera indices to probe with OpenCV (default: 10)",
    )
    parser.add_argument(
        "--kinds",
        nargs="*",
        help=(
            "Optional list of OBS input kinds to include (e.g. dshow_input avcapture_input). "
            "Use 'all' to list every input regardless of kind."
        ),
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        help="Explicit list of camera indices to probe instead of auto-discovery",
    )
    parser.add_argument(
        "--snapshot-dir",
        help="Optional directory to save a single JPEG snapshot per probed camera index.",
    )
    args = parser.parse_args(argv)

    # Load environment variables from an optional .env file.
    if args.dotenv:
        load_dotenv(dotenv_path=Path(args.dotenv).expanduser(), override=False)
    else:
        load_dotenv(override=False)

    password: Optional[str] = args.password
    if password is None:
        candidate_keys: List[str] = []
        if args.password_env:
            candidate_keys.append(args.password_env)
        candidate_keys.extend(["OBS_WEBSOCKET_PASSWORD", "OBS_PASSWORD"])

        seen_keys = set()
        for key in candidate_keys:
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            value = os.getenv(key)
            if value:
                password = value
                break

    kinds_argument: Optional[Sequence[str]]
    if args.kinds and any(kind.lower() in {"all", "*"} for kind in args.kinds):
        kinds_argument = []
    else:
        kinds_argument = args.kinds

    try:
        inputs = fetch_obs_camera_inputs(
            host=args.host,
            port=args.port,
            password=password,
            timeout=args.timeout,
            kinds=kinds_argument,
        )
    except OBSWebSocketError as exc:
        print(f"Error communicating with OBS: {exc}", file=sys.stderr)
        return 2
    except (ConnectionRefusedError, OSError) as exc:
        print(f"Failed to connect to OBS WebSocket ({exc}). Is OBS running with WebSocket enabled?", file=sys.stderr)
        return 2

    if not inputs:
        print("No matching OBS inputs found.")
        return 0

    mappings = map_obs_inputs_to_indices(
        inputs,
        max_devices=args.max_devices,
        preferred_indices=args.indices,
    )

    _print_mapping_table(mappings)

    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
        probed_indices: Set[int] = {mapping.matched_index for mapping in mappings if mapping.matched_index is not None}
        if not probed_indices:
            probed_indices = {mapping.guessed_index for mapping in mappings if mapping.guessed_index is not None}
        if not probed_indices:
            if args.indices:
                probed_indices = set(args.indices)
            else:
                probed_indices = set(list_available_cameras(args.max_devices))

        if probed_indices:
            print(f"Capturing snapshots for indices: {sorted(probed_indices)}")
            saved = capture_snapshots(sorted(probed_indices), snapshot_dir)
            if saved:
                print("Saved snapshots:")
                for path in saved:
                    print(f" - {path}")
            else:
                print("No snapshots were captured.")
        else:
            print("No camera indices available for snapshot capture.")
    return 0


__all__ = [
    "OBSInputInfo",
    "CameraIndexInfo",
    "CameraMapping",
    "OBSWebSocketError",
    "fetch_obs_camera_inputs",
    "probe_camera_indices",
    "map_obs_inputs_to_indices",
]


if __name__ == "__main__":
    sys.exit(main())
