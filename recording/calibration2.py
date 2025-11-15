"""Improved calibration recorder with lighter metadata logging.

Differences vs recording/calibration.py
---------------------------------------
- Camera threads log timestamps + calibration state themselves, so the UI loop
  no longer iterates over every camera each frame. This keeps the pygame loop
  responsive and reduces GIL contention.
- Shared calibration state is maintained once per frame and capture workers
  take atomic snapshots, which eliminates repeated pixel/point calculations.
- Optional preview/rendering behaviour is unchanged, but when previews are
  disabled no frame copies are performed on the main thread.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import pygame

from recording.align import run_alignment_ui
from recording.capture import list_available_cameras
from recording.calibration import (
    CALIBRATION_POINTSETS,
    CalibrationEvent,
    CalibrationPoint,
    CalibrationRecordingResult,
    CalibrationRenderer,
    CalibrationSequencer,
    CameraFrameLog,
    DEFAULT_CALIBRATION_POINTS,
    _blit_frame_to_screen,
    _poll_pygame_events,
    _setup_calibration_display,
    _teardown_calibration_display,
)


# -----------------------------------------------------------------------------
# Shared calibration state snapshot
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibStateSnapshot:
    phase: Literal["countdown", "point", "pause", "finished"]
    point_index: Optional[int]
    point_label: Optional[str]
    normalized: Optional[Tuple[float, float]]
    pixel: Optional[Tuple[int, int]]


class SharedCalibrationState:
    """Thread-safe holder for the current calibration point info."""

    def __init__(self, renderer: CalibrationRenderer) -> None:
        self._renderer = renderer
        self._lock = threading.Lock()
        self._snapshot = CalibStateSnapshot("countdown", None, None, None, None)

    def update(self, state) -> None:
        if state.phase == "point" and state.point is not None:
            norm = tuple(float(v) for v in state.point.normalized_position)
            pixel = self._renderer.normalized_to_pixel(norm)
            snap = CalibStateSnapshot(
                phase="point",
                point_index=state.point_index,
                point_label=state.point.label,
                normalized=norm,
                pixel=pixel,
            )
        else:
            snap = CalibStateSnapshot(state.phase, None, None, None, None)
        with self._lock:
            self._snapshot = snap

    def snapshot(self) -> CalibStateSnapshot:
        with self._lock:
            return self._snapshot


# -----------------------------------------------------------------------------
# Capture session + threading helpers
# -----------------------------------------------------------------------------


@dataclass
class CaptureSession:
    index: int
    capture: cv2.VideoCapture
    writer: Optional[cv2.VideoWriter]
    timestamp_handle: Optional[Path]
    csv_fp: Optional[object]
    window_name: Optional[str] = None
    frame_index: int = 0
    frame_times: List[float] = field(default_factory=list)
    frame_point_indices: List[Optional[int]] = field(default_factory=list)
    frame_point_labels: List[Optional[str]] = field(default_factory=list)
    frame_point_norms: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    frame_point_pixels: List[Optional[Tuple[int, int]]] = field(default_factory=list)
    active: bool = True
    latest_frame: Optional[np.ndarray] = None
    latest_timestamp: float = 0.0
    latest_lock: threading.Lock = field(default_factory=threading.Lock)
    write_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=256))
    capture_thread: Optional[threading.Thread] = None
    writer_thread: Optional[threading.Thread] = None

    def log_frame(self, rel_ts: float, snapshot: CalibStateSnapshot) -> None:
        self.frame_times.append(rel_ts)
        self.frame_point_indices.append(snapshot.point_index)
        self.frame_point_labels.append(snapshot.point_label)
        self.frame_point_norms.append(snapshot.normalized)
        self.frame_point_pixels.append(snapshot.pixel)
        if self.csv_fp is not None:
            csv = self.csv_fp
            idx_val = "" if snapshot.point_index is None else str(snapshot.point_index)
            label_val = snapshot.point_label or ""
            if snapshot.normalized is not None:
                nx = f"{snapshot.normalized[0]:.6f}"
                ny = f"{snapshot.normalized[1]:.6f}"
            else:
                nx = ny = ""
            if snapshot.pixel is not None:
                px = str(int(snapshot.pixel[0]))
                py = str(int(snapshot.pixel[1]))
            else:
                px = py = ""
            csv.write(
                f"{self.frame_index},{rel_ts:.6f},{snapshot.phase},{idx_val},{label_val},{nx},{ny},{px},{py}\n"
            )
        self.frame_index += 1


def _open_camera(
    device_idx: int,
    frame_size: Optional[Tuple[int, int]],
    target_fps: Optional[float],
    fourcc: Optional[str],
) -> cv2.VideoCapture:
    api_prefs = [
        getattr(cv2, "CAP_DSHOW", 700),
        getattr(cv2, "CAP_MSMF", 1400),
        cv2.CAP_ANY,
    ]
    last_error = None
    for api in api_prefs:
        cap = cv2.VideoCapture(device_idx, api)
        if not cap.isOpened():
            cap.release()
            continue
        try:
            if frame_size:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
            if target_fps:
                cap.set(cv2.CAP_PROP_FPS, target_fps)
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            return cap
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            cap.release()
            continue
    raise RuntimeError(f"Failed to open camera {device_idx}: {last_error}")


def _init_writer(path: Path, frame_size: Tuple[int, int], fps: float, codec: str) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer at {path}")
    return writer


def _camera_thread(
    session: CaptureSession,
    *,
    shared_state: SharedCalibrationState,
    start_time: float,
    stop_event: threading.Event,
) -> None:
    try:
        time.sleep(session.index * 0.002)
        while not stop_event.is_set():
            read_ok, frame = session.capture.read()
            if not read_ok:
                warnings.warn(f"Camera {session.index}: read() failed", RuntimeWarning)
                time.sleep(0.01)
                continue
            timestamp = time.monotonic()
            rel_ts = timestamp - start_time
            snapshot = shared_state.snapshot()
            session.log_frame(rel_ts, snapshot)
            with session.latest_lock:
                session.latest_frame = frame
                session.latest_timestamp = timestamp
            try:
                session.write_queue.put((timestamp, frame), block=False)
            except queue.Full:
                try:
                    session.write_queue.get_nowait()
                except queue.Empty:
                    pass
                session.write_queue.put((timestamp, frame), block=False)
    finally:
        session.active = False


def _writer_thread(session: CaptureSession, stop_event: threading.Event) -> None:
    writer = session.writer
    if writer is None:
        return
    while not stop_event.is_set() or not session.write_queue.empty():
        try:
            _, frame = session.write_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        writer.write(frame)


def _start_workers(
    sessions: Iterable[CaptureSession],
    *,
    shared_state: SharedCalibrationState,
    start_time: float,
) -> threading.Event:
    stop_event = threading.Event()
    for session in sessions:
        t = threading.Thread(
            target=_camera_thread,
            args=(session,),
            kwargs={"shared_state": shared_state, "start_time": start_time, "stop_event": stop_event},
            daemon=True,
        )
        session.capture_thread = t
        t.start()
        wt = threading.Thread(target=_writer_thread, args=(session, stop_event), daemon=True)
        session.writer_thread = wt
        wt.start()
    return stop_event


def _stop_workers(sessions: Iterable[CaptureSession], stop_event: threading.Event) -> None:
    stop_event.set()
    for sess in sessions:
        if sess.capture_thread:
            sess.capture_thread.join(timeout=1.0)
        if sess.writer_thread:
            sess.writer_thread.join(timeout=1.0)


# -----------------------------------------------------------------------------
# CLI helpers (frame size/cameras reused from calibration.py)
# -----------------------------------------------------------------------------


def _resolve_frame_size(
    width: Optional[int],
    height: Optional[int],
    *,
    default: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if width is None and height is None:
        return default
    if width is None or height is None:
        raise ValueError("--frame-width and --frame-height must be provided together")
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    return (int(width), int(height))


def _resolve_camera_indices(indices: Sequence[int], auto_discover: bool) -> List[int]:
    if indices:
        return list(indices)
    if auto_discover:
        cams = list_available_cameras()
        if not cams:
            raise RuntimeError("No cameras detected; specify --cameras explicitly.")
        print("Auto-discovered cameras:", " ".join(map(str, cams)))
        return cams
    raise ValueError("No cameras specified. Use --cameras or --auto-discover.")


# -----------------------------------------------------------------------------
# Main recording entry point
# -----------------------------------------------------------------------------


def record_calibration_session2(
    camera_indexes: Sequence[int],
    output_dir: Optional[Path | str],
    *,
    frame_size: Optional[Tuple[int, int]],
    fps: float,
    codec: str,
    camera_fourcc: Optional[str],
    show_preview: bool,
    window_prefix: str,
    preview_scale: float,
    points: Sequence[CalibrationPoint],
    window_size: Tuple[int, int],
    window_position: Optional[Tuple[int, int]],
    point_duration: float,
    pause_duration: float,
    countdown_duration: float,
    fullscreen: bool,
    target_radius: int,
    stop_key: str,
    align_before_start: bool,
    alignment_snapshot: Optional[Path],
    alignment_save: Optional[Path],
    file_extension: Optional[str],
) -> CalibrationRecordingResult:
    if not camera_indexes:
        raise ValueError("camera_indexes must not be empty.")

    sequencer = CalibrationSequencer(
        points,
        point_duration=point_duration,
        pause_duration=pause_duration,
        countdown_duration=countdown_duration,
    )
    renderer = CalibrationRenderer(window_size=window_size, radius=target_radius)
    shared_state = SharedCalibrationState(renderer)

    if fullscreen and window_position is None:
        window_position = (0, 0)

    base_dir = Path(output_dir).expanduser().resolve() if output_dir else Path("data").resolve()
    date_str = time.strftime("%Y%m%d")
    time_str = time.strftime("%H%M%S")
    session_dir = base_dir / date_str / time_str
    session_dir.mkdir(parents=True, exist_ok=True)

    chosen_ext = file_extension if file_extension else (".avi" if codec.upper() == "MJPG" else ".mp4")

    sessions: List[CaptureSession] = []
    video_paths: List[Path] = []
    camera_meta: List[Dict[str, object]] = []

    try:
        for idx in camera_indexes:
            cap = _open_camera(idx, frame_size, fps, camera_fourcc or codec)
            try:
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            except Exception:
                pass
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            writer_fps = reported_fps if reported_fps and 0 < reported_fps < 240 else fps
            out_path = session_dir / f"camera{idx}{chosen_ext}"
            writer = _init_writer(out_path, (actual_w, actual_h), writer_fps, codec)
            video_paths.append(out_path)

            csv_path = session_dir / f"camera{idx}_timestamps.csv"
            csv_fp = csv_path.open("w", encoding="utf-8", newline="")
            csv_fp.write("frame_index,timestamp_sec,phase,point_index,point_label,norm_x,norm_y,pixel_x,pixel_y\n")

            window_name = None
            if show_preview:
                window_name = f"{window_prefix} {idx}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if preview_scale != 1.0:
                    cv2.resizeWindow(window_name, int(actual_w * preview_scale), int(actual_h * preview_scale))

            sessions.append(
                CaptureSession(
                    index=idx,
                    capture=cap,
                    writer=writer,
                    timestamp_handle=csv_path,
                    csv_fp=csv_fp,
                    window_name=window_name,
                )
            )
            camera_meta.append(
                {
                    "index": idx,
                    "video_path": str(out_path),
                    "timestamp_csv": str(csv_path),
                    "actual_width": actual_w,
                    "actual_height": actual_h,
                    "requested_fps": fps,
                    "driver_reported_fps": reported_fps,
                    "writer_fps": writer_fps,
                }
            )
    except Exception:
        for sess in sessions:
            sess.capture.release()
            if sess.writer:
                sess.writer.release()
        raise

    if align_before_start:
        try:
            camera_ids = [sess.index for sess in sessions]
            alignment_result = run_alignment_ui(
                camera_ids,
                labels=[f"Camera {idx}" for idx in camera_ids],
                canvas_size=None,
                snapshot_path=alignment_snapshot,
                output_path=alignment_save,
            )
            print("Alignment positions:")
            for cam_id in camera_ids:
                key = str(cam_id)
                if key in alignment_result:
                    x, y = alignment_result[key]
                    print(f"  camera {cam_id}: ({x:.3f}, {y:.3f})")
        except RuntimeError as exc:
            print(f"Warning: alignment UI failed: {exc}", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: unexpected alignment error: {exc}", file=sys.stderr)

    pygame_screen = None
    try:
        pygame_screen, renderer = _setup_calibration_display(
            calibration_window_name="Calibration",
            renderer=renderer,
            fullscreen=fullscreen,
            window_position=window_position,
        )
    except Exception:
        for sess in sessions:
            sess.capture.release()
        raise

    meta = {
        "session_dir": str(session_dir),
        "timestamp": f"{date_str}/{time_str}",
        "cameras": camera_meta,
        "calibration": {
            "points": [
                {
                    "label": p.label,
                    "normalized": {"x": p.normalized_position[0], "y": p.normalized_position[1]},
                }
                for p in points
            ],
            "durations": {
                "point": point_duration,
                "pause": pause_duration,
                "countdown": countdown_duration,
                "total": sequencer.total_duration,
            },
        },
    }
    (session_dir / "session_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    start_time = time.monotonic()
    stop_event = _start_workers(sessions, shared_state=shared_state, start_time=start_time)

    point_records: List[_PointRecord] = []
    active_point_index: Optional[int] = None
    skip_trigger = {"flag": False}
    aborted = False
    manual_offset = 0.0

    try:
        pygame_clock = pygame.time.Clock()
        while True:
            elapsed = time.monotonic() - start_time
            adjusted_elapsed = elapsed + manual_offset
            state = sequencer.state_for_elapsed(adjusted_elapsed)
            stage_start_real = max(0.0, elapsed - state.elapsed)
            shared_state.update(state)

            skip, abort = _poll_pygame_events(ord(stop_key) if stop_key else None)
            if abort:
                aborted = True
                break
            if skip:
                skip_trigger["flag"] = True

            if skip_trigger["flag"] and state.phase != "finished":
                if state.phase == "point" and active_point_index is not None and point_records:
                    last = point_records[-1]
                    if last.end_time is None:
                        last.end_time = stage_start_real
                active_point_index = None
                manual_offset += max(0.0, state.remaining) + 1e-3
                skip_trigger["flag"] = False
                continue
            skip_trigger["flag"] = False

            if state.phase == "point" and state.point is not None:
                if active_point_index != state.point_index:
                    if active_point_index is not None and point_records:
                        last = point_records[-1]
                        if last.end_time is None:
                            last.end_time = stage_start_real
                    active_point_index = state.point_index
                    pixel_pos = renderer.normalized_to_pixel(state.point.normalized_position)
                    point_records.append(
                        _PointRecord(
                            point_index=state.point_index,
                            label=state.point.label,
                            start_time=stage_start_real,
                            normalized_position=state.point.normalized_position,
                            pixel_position=pixel_pos,
                        )
                    )
            else:
                if active_point_index is not None and point_records:
                    last = point_records[-1]
                    if last.end_time is None:
                        last.end_time = stage_start_real
                active_point_index = None

            frame_ui = renderer.render(state)
            _blit_frame_to_screen(pygame_screen, frame_ui)

            if show_preview:
                for sess in sessions:
                    with sess.latest_lock:
                        frame = sess.latest_frame
                    if frame is not None and sess.window_name:
                        preview = frame
                        if preview_scale != 1.0:
                            preview = cv2.resize(preview, (0, 0), fx=preview_scale, fy=preview_scale)
                        cv2.imshow(sess.window_name, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break

            if state.phase == "finished":
                break

            pygame_clock.tick(60)
    finally:
        _stop_workers(sessions, stop_event)
        _teardown_calibration_display()

    events: List[CalibrationEvent] = []
    total_duration = time.monotonic() - start_time
    for rec in point_records:
        events.append(
            CalibrationEvent(
                point_index=rec.point_index,
                label=rec.label,
                start_time=rec.start_time,
                end_time=rec.end_time or total_duration,
                normalized_position=rec.normalized_position,
                pixel_position=rec.pixel_position,
            )
        )

    frame_logs: List[CameraFrameLog] = []
    for sess in sessions:
        if sess.writer:
            sess.writer.release()
        if sess.capture.isOpened():
            sess.capture.release()
        if sess.csv_fp:
            sess.csv_fp.close()
        if sess.window_name:
            try:
                cv2.destroyWindow(sess.window_name)
            except Exception:
                pass
        frame_logs.append(
            CameraFrameLog(
                camera_index=sess.index,
                timestamps=tuple(sess.frame_times),
                point_indices=tuple(sess.frame_point_indices),
                point_labels=tuple(sess.frame_point_labels),
                normalized_positions=tuple(sess.frame_point_norms),
                pixel_positions=tuple(sess.frame_point_pixels),
                csv_path=sess.timestamp_handle,
            )
        )

    return CalibrationRecordingResult(
        video_paths=tuple(video_paths),
        events=tuple(events),
        frame_logs=tuple(frame_logs),
        aborted=aborted,
        duration=total_duration,
    )


@dataclass
class _PointRecord:
    point_index: int
    label: str
    start_time: float
    end_time: Optional[float] = None
    normalized_position: Tuple[float, float] = (0.0, 0.0)
    pixel_position: Tuple[int, int] = (0, 0)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alternative calibration recorder with lighter logging.")
    parser.add_argument("--output-dir", help="Directory where recordings will be stored.")
    parser.add_argument("--cameras", nargs="*", type=int, metavar="INDEX")
    parser.add_argument("--auto-discover", action="store_true", help="Use all detected cameras when --cameras omitted.")
    parser.add_argument("--frame-width", type=int, help="Capture width (requires --frame-height).")
    parser.add_argument("--frame-height", type=int, help="Capture height (requires --frame-width).")
    parser.add_argument("--native-resolution", action="store_true", help="Use native camera resolution.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS per camera.")
    parser.add_argument("--codec", default="MJPG", help="VideoWriter codec (default: MJPG).")
    parser.add_argument("--camera-fourcc", help="FourCC requested for camera capture (default: same as codec).")
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--window-prefix", default="Camera")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--window-x", type=int)
    parser.add_argument("--window-y", type=int)
    parser.add_argument("--point-duration", type=float, default=2.0)
    parser.add_argument("--pause-duration", type=float, default=0.6)
    parser.add_argument("--countdown-duration", type=float, default=1.5)
    parser.add_argument("--target-radius", type=int, default=18)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--file-extension", help="Override output file extension.")
    parser.add_argument("--stop-key", default="q")
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--align-snapshot", type=Path)
    parser.add_argument("--align-save", type=Path)
    parser.add_argument(
        "--pointset",
        type=int,
        choices=sorted(CALIBRATION_POINTSETS.keys()),
        default=1,
        help="Which predefined point set to use (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    default_frame = None if args.native_resolution else (1920, 1080)
    try:
        frame_size = _resolve_frame_size(args.frame_width, args.frame_height, default=default_frame)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        camera_indices = _resolve_camera_indices(args.cameras or [], args.auto_discover)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    window_pos = None
    if args.window_x is not None and args.window_y is not None:
        window_pos = (args.window_x, args.window_y)

    points = CALIBRATION_POINTSETS.get(args.pointset, DEFAULT_CALIBRATION_POINTS)

    try:
        result = record_calibration_session2(
            camera_indices,
            args.output_dir,
            frame_size=frame_size,
            fps=args.fps,
            codec=args.codec,
            camera_fourcc=args.camera_fourcc or args.codec,
            show_preview=args.show_preview,
            window_prefix=args.window_prefix,
            preview_scale=args.preview_scale,
            points=points,
            window_size=(args.window_width, args.window_height),
            window_position=window_pos,
            point_duration=args.point_duration,
            pause_duration=args.pause_duration,
            countdown_duration=args.countdown_duration,
            fullscreen=not args.windowed,
            target_radius=args.target_radius,
            stop_key=args.stop_key,
            align_before_start=args.align,
            alignment_snapshot=args.align_snapshot,
            alignment_save=args.align_save,
            file_extension=args.file_extension,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Saved recordings:")
    for path in result.video_paths:
        print(" -", path)

    print("Frame timestamp logs:")
    for log in result.frame_logs:
        fps = _estimate_fps(log.timestamps)
        fps_str = f"{fps:.2f}" if fps else "unknown"
        location = log.csv_path or "(not written)"
        print(f" - camera {log.camera_index}: {len(log.timestamps)} frames ({fps_str} fps), timestamps -> {location}")

    if result.aborted:
        print("Session ended early (stop key).")
    return 0


def _estimate_fps(timestamps: Tuple[float, ...]) -> Optional[float]:
    if len(timestamps) < 2:
        return None
    diffs = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not diffs:
        return None
    avg = sum(diffs) / len(diffs)
    if avg <= 0:
        return None
    return 1.0 / avg


if __name__ == "__main__":
    raise SystemExit(main())
