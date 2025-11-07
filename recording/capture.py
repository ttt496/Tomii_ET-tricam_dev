"""Helpers for recording simultaneous webcam streams."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2


@dataclass
class _CameraSession:
    index: int
    capture: cv2.VideoCapture
    writer: Optional[cv2.VideoWriter]
    window_name: Optional[str]
    output_path: Optional[Path]
    active: bool = True


def list_available_cameras(max_devices: int = 10) -> List[int]:
    """Return a list of device indices that appear to be valid webcams.

    Windows 環境ではバックエンドによって初期化可否が変わるため、
    DirectShow → MSMF → 自動 の順で試行して検出精度を上げる。
    """
    found: List[int] = []
    api_preferences = [
        getattr(cv2, "CAP_DSHOW", 700),  # DirectShow (定数が無いOpenCVでも数値で可)
        getattr(cv2, "CAP_MSMF", 1400),  # Media Foundation
        getattr(cv2, "CAP_ANY", 0),
    ]
    for device_idx in range(max_devices):
        detected = False
        for api_pref in api_preferences:
            cap = cv2.VideoCapture(device_idx, api_pref)
            if not cap.isOpened():
                cap.release()
                continue
            success, _ = cap.read()
            cap.release()
            if success:
                detected = True
                break
        if detected:
            found.append(device_idx)
    return found


def _open_camera(device_idx: int, frame_size: Optional[Tuple[int, int]]) -> cv2.VideoCapture:
    """Open a camera, trying multiple API backends for robustness on Windows."""
    api_preferences = [
        getattr(cv2, "CAP_DSHOW", 700),
        getattr(cv2, "CAP_MSMF", 1400),
        getattr(cv2, "CAP_ANY", 0),
    ]
    last_error: Optional[str] = None
    for api_pref in api_preferences:
        cap = cv2.VideoCapture(device_idx, api_pref)
        if not cap.isOpened():
            cap.release()
            continue

        if frame_size is not None:
            width, height = frame_size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        success, _ = cap.read()
        if success:
            return cap
        last_error = "no initial frame"
        cap.release()

    raise RuntimeError(f"Failed to open camera {device_idx}: {last_error or 'unavailable'}")


def _init_writer(output_dir: Path, device_idx: int, frame_size: Tuple[int, int], fps: float, codec: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_path = output_dir / f"camera_{device_idx}.mp4"
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for camera {device_idx}")
    return writer


def _close_session(session: _CameraSession) -> None:
    if not session.active:
        return

    if session.capture.isOpened():
        session.capture.release()

    if session.writer is not None and session.writer.isOpened():
        session.writer.release()
        session.writer = None

    if session.window_name is not None:
        cv2.destroyWindow(session.window_name)
        session.window_name = None

    session.active = False


def record_from_cameras(
    camera_indexes: Sequence[int],
    output_dir: Optional[Path | str] = None,
    *,
    frame_size: Optional[Tuple[int, int]] = None,
    fps: float = 30.0,
    codec: str = "mp4v",
    show_preview: bool = True,
    window_prefix: str = "Camera",
    preview_scale: float = 1.0,
    max_seconds: Optional[float] = None,
) -> List[Path]:
    """Record one or more webcams simultaneously.

    Parameters
    ----------
    camera_indexes:
        Device indices to record.
    output_dir:
        Directory to store recordings. If ``None`` no files are written.
    frame_size:
        Optional target frame size as ``(width, height)``. The camera's native
        resolution is used when omitted.
    fps:
        Target frames per second for the output recordings.
    codec:
        Four-character OpenCV codec string.
    show_preview:
        When true, display each stream in an on-screen window.
    window_prefix:
        Prefix used when naming preview windows.
    preview_scale:
        Scaling factor for preview windows. ``1.0`` keeps the original size.
    max_seconds:
        Optional limit on recording duration.

    Returns
    -------
    list[Path]
        Paths to generated video files. Empty when ``output_dir`` is ``None``.

    Raises
    ------
    RuntimeError
        If any of the cameras cannot be initialised or recordings fail to start.
    """
    if not camera_indexes:
        raise ValueError("camera_indexes must contain at least one device index")

    sessions: List[_CameraSession] = []
    output_paths: List[Path] = []

    output_directory: Optional[Path] = Path(output_dir).expanduser().resolve() if output_dir else None
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)

    frames_captured = False

    try:
        for idx in camera_indexes:
            cap = _open_camera(idx, frame_size)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer: Optional[cv2.VideoWriter] = None
            output_path: Optional[Path] = None
            if output_directory is not None:
                writer = _init_writer(output_directory, idx, (actual_width, actual_height), fps, codec)
                output_path = output_directory / f"camera_{idx}.mp4"
                output_paths.append(output_path)

            window_name: Optional[str] = None
            if show_preview:
                window_name = f"{window_prefix} {idx}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if preview_scale != 1.0:
                    cv2.resizeWindow(window_name, int(actual_width * preview_scale), int(actual_height * preview_scale))

            sessions.append(_CameraSession(idx, cap, writer, window_name, output_path))

        start_time = time.monotonic()
        while sessions:
            for session in list(sessions):
                success, frame = session.capture.read()
                if not success:
                    print(f"Warning: Camera {session.index} stopped delivering frames; closing stream.", file=sys.stderr)
                    _close_session(session)
                    sessions.remove(session)
                    continue

                frames_captured = True

                if session.writer is not None:
                    session.writer.write(frame)

                if session.window_name is not None:
                    cv2.imshow(session.window_name, frame)

            if not sessions:
                break

            if show_preview:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_seconds is not None and (time.monotonic() - start_time) >= max_seconds:
                break

        if not sessions and not frames_captured:
            raise RuntimeError("All cameras stopped delivering frames immediately.")

    except KeyboardInterrupt:
        pass
    finally:
        for session in sessions:
            _close_session(session)
        if show_preview:
            cv2.waitKey(1)
            cv2.destroyAllWindows()

    return output_paths
