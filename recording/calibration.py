"""Calibration UI utilities for multi-camera gaze data capture."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, TextIO, Tuple

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from recording.capture import list_available_cameras
    from recording.align import run_alignment_ui
else:
    from .capture import list_available_cameras
    from .align import run_alignment_ui


@dataclass(frozen=True)
class CalibrationPoint:
    """A single calibration target defined in normalised (0..1) coordinates."""

    label: str
    normalized_position: Tuple[float, float]


DEFAULT_CALIBRATION_POINTS: Tuple[CalibrationPoint, ...] = (
    CalibrationPoint("top_left", (0.2, 0.2)),
    CalibrationPoint("top_center", (0.5, 0.2)),
    CalibrationPoint("top_right", (0.8, 0.2)),
    CalibrationPoint("middle_left", (0.2, 0.5)),
    CalibrationPoint("middle_center", (0.5, 0.5)),
    CalibrationPoint("middle_right", (0.8, 0.5)),
    CalibrationPoint("bottom_left", (0.2, 0.8)),
    CalibrationPoint("bottom_center", (0.5, 0.8)),
    CalibrationPoint("bottom_right", (0.8, 0.8)),
)


@dataclass(frozen=True)
class CalibrationPhaseState:
    """Describes the UI phase at a particular elapsed time."""

    phase: Literal["countdown", "point", "pause", "finished"]
    point_index: Optional[int]
    point: Optional[CalibrationPoint]
    elapsed: float
    remaining: float
    total_elapsed: float
    total_duration: float


class CalibrationSequencer:
    """Track the progression through calibration targets over time."""

    def __init__(
        self,
        points: Sequence[CalibrationPoint] = DEFAULT_CALIBRATION_POINTS,
        *,
        point_duration: float = 2.0,
        pause_duration: float = 0.6,
        countdown_duration: float = 1.5,
    ) -> None:
        if not points:
            raise ValueError("points must contain at least one calibration point")
        if point_duration <= 0:
            raise ValueError("point_duration must be positive")
        if pause_duration < 0:
            raise ValueError("pause_duration cannot be negative")
        if countdown_duration < 0:
            raise ValueError("countdown_duration cannot be negative")

        self._points: Tuple[CalibrationPoint, ...] = tuple(points)
        self._point_duration = float(point_duration)
        self._pause_duration = float(pause_duration)
        self._countdown_duration = float(countdown_duration)

        pauses = max(0, len(self._points) - 1)
        self._total_duration = (
            self._countdown_duration
            + len(self._points) * self._point_duration
            + pauses * self._pause_duration
        )

    @property
    def points(self) -> Tuple[CalibrationPoint, ...]:
        return self._points

    @property
    def total_duration(self) -> float:
        return self._total_duration

    @property
    def point_duration(self) -> float:
        return self._point_duration

    @property
    def pause_duration(self) -> float:
        return self._pause_duration

    @property
    def countdown_duration(self) -> float:
        return self._countdown_duration

    def state_for_elapsed(self, elapsed_seconds: float) -> CalibrationPhaseState:
        """Return the calibration phase for a given elapsed time."""
        if elapsed_seconds < 0:
            elapsed_seconds = 0.0

        if elapsed_seconds < self._countdown_duration:
            remaining = self._countdown_duration - elapsed_seconds
            return CalibrationPhaseState(
                phase="countdown",
                point_index=None,
                point=None,
                elapsed=elapsed_seconds,
                remaining=remaining,
                total_elapsed=elapsed_seconds,
                total_duration=self._total_duration,
            )

        elapsed = elapsed_seconds - self._countdown_duration
        for idx, point in enumerate(self._points):
            if elapsed < self._point_duration:
                remaining = self._point_duration - elapsed
                return CalibrationPhaseState(
                    phase="point",
                    point_index=idx,
                    point=point,
                    elapsed=elapsed,
                    remaining=remaining,
                    total_elapsed=elapsed_seconds,
                    total_duration=self._total_duration,
                )
            elapsed -= self._point_duration

            if idx == len(self._points) - 1:
                break

            if self._pause_duration > 0:
                if elapsed < self._pause_duration:
                    remaining = self._pause_duration - elapsed
                    return CalibrationPhaseState(
                        phase="pause",
                        point_index=idx,
                        point=None,
                        elapsed=elapsed,
                        remaining=remaining,
                        total_elapsed=elapsed_seconds,
                        total_duration=self._total_duration,
                    )
                elapsed -= self._pause_duration

        return CalibrationPhaseState(
            phase="finished",
            point_index=None,
            point=None,
            elapsed=self._total_duration,
            remaining=0.0,
            total_elapsed=min(elapsed_seconds, self._total_duration),
            total_duration=self._total_duration,
        )


class CalibrationRenderer:
    """Render calibration phases onto frames."""

    def __init__(
        self,
        *,
        window_size: Tuple[int, int] = (1980, 1080),
        background_color: Tuple[int, int, int] = (25, 25, 25),
        target_color: Tuple[int, int, int] = (40, 200, 255),
        pause_color: Tuple[int, int, int] = (80, 80, 80),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        radius: int = 18,
        thickness: int = -1,
        font_scale: float = 1.0,
    ) -> None:
        width, height = window_size
        if width <= 0 or height <= 0:
            raise ValueError("window_size must contain positive dimensions")
        self._width = int(width)
        self._height = int(height)
        self._background_color = tuple(int(c) for c in background_color)
        self._target_color = tuple(int(c) for c in target_color)
        self._pause_color = tuple(int(c) for c in pause_color)
        self._text_color = tuple(int(c) for c in text_color)
        self._radius = int(radius)
        self._thickness = int(thickness)
        self._font_scale = float(font_scale)

    @property
    def window_size(self) -> Tuple[int, int]:
        return self._width, self._height

    def normalized_to_pixel(self, position: Tuple[float, float]) -> Tuple[int, int]:
        x_norm, y_norm = position
        x = int(round(x_norm * (self._width - 1)))
        y = int(round(y_norm * (self._height - 1)))
        return x, y

    def _draw_skip_hint(self, frame: np.ndarray) -> None:
        cv2.putText(
            frame,
            "Click or press Space to skip",
            (60, self._height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._font_scale * 0.7,
            self._text_color,
            2,
            cv2.LINE_AA,
        )

    def render(self, state: CalibrationPhaseState) -> np.ndarray:
        """Render a frame for the given calibration state."""
        frame = np.full((self._height, self._width, 3), self._background_color, dtype=np.uint8)

        if state.phase == "point" and state.point is not None:
            x, y = self.normalized_to_pixel(state.point.normalized_position)
            cv2.circle(
                frame,
                (x, y),
                self._radius,
                self._target_color,
                self._thickness,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (x, y),
                max(2, self._radius // 3),
                (0, 0, 0),
                self._thickness,
                cv2.LINE_AA,
            )
            return frame

        return frame

    def _draw_progress(self, frame: np.ndarray, state: CalibrationPhaseState) -> None:
        if state.total_duration <= 0:
            return
        progress_ratio = min(1.0, max(0.0, state.total_elapsed / state.total_duration))
        bar_width = int(self._width * 0.6)
        bar_height = 16
        start_x = (self._width - bar_width) // 2
        start_y = self._height - 60
        cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), self._pause_color, 2)
        fill_width = int(bar_width * progress_ratio)
        cv2.rectangle(
            frame,
            (start_x, start_y),
            (start_x + fill_width, start_y + bar_height),
            self._target_color,
            -1,
        )


@dataclass(frozen=True)
class CalibrationEvent:
    """Timing metadata for a single calibration point during capture."""

    point_index: int
    label: str
    start_time: float
    end_time: float
    normalized_position: Tuple[float, float]
    pixel_position: Tuple[int, int]


@dataclass(frozen=True)
class CalibrationRecordingResult:
    """Aggregate result returned after a calibration capture session."""

    video_paths: Tuple[Path, ...]
    events: Tuple[CalibrationEvent, ...]
    frame_logs: Tuple["CameraFrameLog", ...]
    aborted: bool
    duration: float


@dataclass
class _CalibrationSession:
    index: int
    capture: cv2.VideoCapture
    writer: Optional[cv2.VideoWriter]
    window_name: Optional[str]
    output_path: Optional[Path]
    timestamp_path: Optional[Path]
    timestamp_handle: Optional[TextIO]
    frame_index: int = 0
    frame_times: List[float] = field(default_factory=list)
    frame_point_indices: List[Optional[int]] = field(default_factory=list)
    frame_point_labels: List[Optional[str]] = field(default_factory=list)
    frame_point_norms: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    frame_point_pixels: List[Optional[Tuple[int, int]]] = field(default_factory=list)
    active: bool = True


@dataclass
class _PointRecord:
    point_index: int
    label: str
    start_time: float
    end_time: Optional[float] = None
    normalized_position: Tuple[float, float] = (0.0, 0.0)
    pixel_position: Tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class CameraFrameLog:
    camera_index: int
    timestamps: Tuple[float, ...]
    point_indices: Tuple[Optional[int], ...]
    point_labels: Tuple[Optional[str], ...]
    normalized_positions: Tuple[Optional[Tuple[float, float]], ...]
    pixel_positions: Tuple[Optional[Tuple[int, int]], ...]
    csv_path: Optional[Path]


def _open_camera(
    device_idx: int,
    frame_size: Optional[Tuple[int, int]],
    target_fps: Optional[float],
    fourcc: Optional[str],
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {device_idx}")

    if frame_size is not None:
        width, height = frame_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    if target_fps is not None and target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))

    if fourcc:
        try:
            code = fourcc.upper()
            if len(code) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        except cv2.error:
            pass

    success, _ = cap.read()
    if not success:
        cap.release()
        raise RuntimeError(f"Camera {device_idx} failed to provide an initial frame")
    return cap


def _init_writer(
    output_dir: Path,
    device_idx: int,
    frame_size: Tuple[int, int],
    fps: float,
    codec: str,
    extension: str,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_path = output_dir / f"camera_{device_idx}{extension}"
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for camera {device_idx}")
    return writer


def _close_session(session: _CalibrationSession) -> None:
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

    if session.timestamp_handle is not None:
        session.timestamp_handle.close()
        session.timestamp_handle = None

    session.active = False


def record_calibration_session(
    camera_indexes: Sequence[int],
    output_dir: Optional[Path | str] = None,
    *,
    frame_size: Optional[Tuple[int, int]] = None,
    fps: float = 30.0,
    codec: str = "MJPG",
    camera_fourcc: Optional[str] = "MJPG",
    show_preview: bool = False,
    window_prefix: str = "Camera",
    preview_scale: float = 1.0,
    points: Optional[Sequence[CalibrationPoint]] = None,
    window_size: Tuple[int, int] = (1280, 720),
    point_duration: float = 2.0,
    pause_duration: float = 0.6,
    countdown_duration: float = 1.5,
    calibration_window_name: str = "Calibration",
    fullscreen: bool = True,
    window_position: Optional[Tuple[int, int]] = None,
    target_radius: int = 18,
    stop_key: str = "q",
    align_before_start: bool = False,
    alignment_snapshot: Optional[Path | str] = None,
    alignment_save: Optional[Path | str] = None,
    file_extension: Optional[str] = None,
) -> CalibrationRecordingResult:
    """Record from multiple cameras while showing sequential calibration targets."""

    if not camera_indexes:
        raise ValueError("camera_indexes must contain at least one device index")

    point_sequence = tuple(points) if points else DEFAULT_CALIBRATION_POINTS
    sequencer = CalibrationSequencer(
        point_sequence,
        point_duration=point_duration,
        pause_duration=pause_duration,
        countdown_duration=countdown_duration,
    )
    renderer = CalibrationRenderer(window_size=window_size, radius=target_radius)

    chosen_extension: str
    if file_extension:
        chosen_extension = file_extension if file_extension.startswith(".") else f".{file_extension}"
    else:
        chosen_extension = ".avi" if codec.upper() == "MJPG" else ".mp4"

    alignment_snapshot_path: Optional[Path] = None
    alignment_save_path: Optional[Path] = None
    if align_before_start:
        if alignment_snapshot is not None:
            alignment_snapshot_path = Path(alignment_snapshot).expanduser().resolve()
        if alignment_save is not None:
            alignment_save_path = Path(alignment_save).expanduser().resolve()

    cv2.namedWindow(calibration_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(calibration_window_name, *renderer.window_size)
    if window_position is not None:
        cv2.moveWindow(
            calibration_window_name,
            int(window_position[0]),
            int(window_position[1]),
        )
    if fullscreen:
        cv2.setWindowProperty(
            calibration_window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        try:
            _, _, w, h = cv2.getWindowImageRect(calibration_window_name)
            if w > 0 and h > 0:
                renderer = CalibrationRenderer(window_size=(w, h), radius=target_radius)
        except cv2.error:
            pass
        blank = np.zeros((renderer.window_size[1], renderer.window_size[0], 3), dtype=np.uint8)
        cv2.imshow(calibration_window_name, blank)
        cv2.waitKey(1)

    skip_trigger = {"flag": False}

    def _calibration_mouse_handler(event: int, x: int, y: int, flags: int, param: Optional[int]) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            skip_trigger["flag"] = True

    cv2.setMouseCallback(calibration_window_name, _calibration_mouse_handler)

    output_directory: Optional[Path] = None
    if output_dir is not None:
        output_directory = Path(output_dir).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

    sessions: List[_CalibrationSession] = []
    all_sessions: List[_CalibrationSession] = []
    video_paths: List[Path] = []

    try:
        for idx in camera_indexes:
            cap = _open_camera(idx, frame_size, fps, camera_fourcc)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer: Optional[cv2.VideoWriter] = None
            output_path: Optional[Path] = None
            timestamp_path: Optional[Path] = None
            timestamp_handle: Optional[TextIO] = None
            if output_directory is not None:
                writer = _init_writer(output_directory, idx, (actual_width, actual_height), fps, codec, chosen_extension)
                output_path = output_directory / f"camera_{idx}{chosen_extension}"
                video_paths.append(output_path)

                timestamp_path = output_directory / f"camera_{idx}_timestamps.csv"
                timestamp_handle = timestamp_path.open("w", encoding="utf-8", newline="")
                timestamp_handle.write(
                    "frame_index,timestamp_sec,phase,point_index,point_label,norm_x,norm_y,pixel_x,pixel_y\n"
                )

            window_name: Optional[str] = None
            if show_preview:
                window_name = f"{window_prefix} {idx}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if preview_scale != 1.0:
                    cv2.resizeWindow(
                        window_name,
                        int(actual_width * preview_scale),
                        int(actual_height * preview_scale),
                    )

            session = _CalibrationSession(
                idx,
                cap,
                writer,
                window_name,
                output_path,
                timestamp_path,
                timestamp_handle,
            )
            sessions.append(session)
            all_sessions.append(session)

    except Exception:
        for session in all_sessions:
            _close_session(session)
        cv2.destroyWindow(calibration_window_name)
        cv2.waitKey(1)
        raise

    # Launch optional alignment UI
    if align_before_start:
        try:
            camera_ids = [session.index for session in sessions]
            alignment_result = run_alignment_ui(
                camera_ids,
                labels=[f"Camera {idx}" for idx in camera_ids],
                canvas_size=None,
                snapshot_path=alignment_snapshot_path,
                output_path=alignment_save_path,
            )
            print("Alignment positions (normalized coordinates):")
            for cam_id in camera_ids:
                key = str(cam_id)
                if key in alignment_result:
                    x, y = alignment_result[key]
                    print(f" - camera {cam_id}: ({x:.4f}, {y:.4f})")
        except RuntimeError as exc:
            print(f"Warning: alignment UI failed to run: {exc}", file=sys.stderr)
        except Exception as exc:  # safety catch
            print(f"Warning: unexpected alignment UI error: {exc}", file=sys.stderr)

    frames_captured = False
    aborted = False
    point_records: List[_PointRecord] = []
    active_point_index: Optional[int] = None
    start_time = time.monotonic()
    manual_offset = 0.0

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            adjusted_elapsed = elapsed + manual_offset
            state = sequencer.state_for_elapsed(adjusted_elapsed)
            stage_start_real = max(0.0, elapsed - state.elapsed)

            if skip_trigger["flag"] and state.phase != "finished":
                if state.phase == "point" and active_point_index is not None and point_records:
                    last = point_records[-1]
                    if last.end_time is None:
                        last.end_time = elapsed
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
                    normalized_position = state.point.normalized_position
                    pixel_position = renderer.normalized_to_pixel(normalized_position)
                    point_records.append(
                        _PointRecord(
                            point_index=state.point_index,
                            label=state.point.label,
                            start_time=stage_start_real,
                            normalized_position=normalized_position,
                            pixel_position=pixel_position,
                        )
                    )
            else:
                if active_point_index is not None and point_records:
                    last = point_records[-1]
                    if last.end_time is None:
                        last.end_time = stage_start_real
                active_point_index = None

            point_idx_for_log: Optional[int]
            point_label_for_log: Optional[str]
            norm_for_log: Optional[Tuple[float, float]]
            pixel_for_log: Optional[Tuple[int, int]]
            if state.phase == "point" and state.point is not None:
                point_idx_for_log = state.point_index
                point_label_for_log = state.point.label
                norm_for_log = tuple(float(v) for v in state.point.normalized_position)
                pixel_for_log = renderer.normalized_to_pixel(state.point.normalized_position)
            else:
                point_idx_for_log = None
                point_label_for_log = None
                norm_for_log = None
                pixel_for_log = None

            for session in list(sessions):
                success, frame = session.capture.read()
                if not success:
                    print(
                        f"Warning: Camera {session.index} stopped delivering frames; closing stream.",
                        file=sys.stderr,
                    )
                    _close_session(session)
                    sessions.remove(session)
                    continue

                frames_captured = True

                frame_timestamp = time.monotonic() - start_time
                session.frame_times.append(frame_timestamp)
                session.frame_point_indices.append(point_idx_for_log)
                session.frame_point_labels.append(point_label_for_log)
                session.frame_point_norms.append(norm_for_log)
                session.frame_point_pixels.append(pixel_for_log)

                frame_index = session.frame_index
                session.frame_index += 1

                if session.timestamp_handle is not None:
                    phase_value = state.phase
                    idx_value = "" if point_idx_for_log is None else str(point_idx_for_log)
                    label_value = "" if point_label_for_log is None else point_label_for_log
                    if norm_for_log is not None:
                        norm_x, norm_y = norm_for_log
                        norm_x_str = f"{norm_x:.6f}"
                        norm_y_str = f"{norm_y:.6f}"
                        if pixel_for_log is not None:
                            pixel_x, pixel_y = pixel_for_log
                            pixel_x_str = str(int(pixel_x))
                            pixel_y_str = str(int(pixel_y))
                        else:
                            pixel_x_str = pixel_y_str = ""
                    else:
                        norm_x_str = norm_y_str = ""
                        pixel_x_str = pixel_y_str = ""
                    session.timestamp_handle.write(
                        f"{frame_index},{frame_timestamp:.6f},{phase_value},{idx_value},{label_value},{norm_x_str},{norm_y_str},{pixel_x_str},{pixel_y_str}\n"
                    )

                if session.writer is not None:
                    session.writer.write(frame)

                if show_preview and session.window_name is not None:
                    cv2.imshow(session.window_name, frame)

            if not sessions:
                raise RuntimeError("All cameras stopped delivering frames.")

            calibration_frame = renderer.render(state)
            cv2.imshow(calibration_window_name, calibration_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(stop_key) or key == 27:
                aborted = True
                break

            if key in (ord(" "), ord("\r"), ord("\n")):
                skip_trigger["flag"] = True
                continue

            if state.phase == "finished":
                break

    except KeyboardInterrupt:
        aborted = True
    finally:
        for session in all_sessions:
            _close_session(session)
        cv2.destroyWindow(calibration_window_name)
        cv2.waitKey(1)

    final_elapsed = time.monotonic() - start_time
    if point_records and point_records[-1].end_time is None:
        point_records[-1].end_time = final_elapsed

    if not frames_captured:
        raise RuntimeError("No frames captured during calibration session.")

    events: List[CalibrationEvent] = []
    for record in point_records:
        start_time_value = float(record.start_time)
        end_time_value = float(record.end_time) if record.end_time is not None else final_elapsed
        events.append(
            CalibrationEvent(
                point_index=int(record.point_index),
                label=str(record.label),
                start_time=start_time_value,
                end_time=end_time_value,
                normalized_position=tuple(float(v) for v in record.normalized_position),
                pixel_position=tuple(int(v) for v in record.pixel_position),
            )
        )

    frame_logs: List[CameraFrameLog] = []
    for session in all_sessions:
        frame_logs.append(
            CameraFrameLog(
                camera_index=session.index,
                timestamps=tuple(session.frame_times),
                point_indices=tuple(session.frame_point_indices),
                point_labels=tuple(session.frame_point_labels),
                normalized_positions=tuple(session.frame_point_norms),
                pixel_positions=tuple(session.frame_point_pixels),
                csv_path=session.timestamp_path,
            )
        )

    return CalibrationRecordingResult(
        video_paths=tuple(video_paths),
        events=tuple(events),
        frame_logs=tuple(frame_logs),
        aborted=aborted,
        duration=final_elapsed,
    )


def _resolve_frame_size(width: Optional[int], height: Optional[int]) -> Optional[Tuple[int, int]]:
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("--frame-width and --frame-height must be provided together")
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    return (int(width), int(height))


def _resolve_camera_indices(indices: Sequence[int], auto_discover: bool) -> List[int]:
    if indices:
        return list(indices)

    if auto_discover:
        cameras = list_available_cameras()
        if not cameras:
            raise RuntimeError("No cameras detected. Specify --cameras explicitly.")
        print("Auto-discovered cameras:", " ".join(map(str, cameras)))
        return cameras

    raise ValueError("No cameras specified. Use --cameras or --auto-discover.")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record multi-camera calibration data with 9-point targets.")
    parser.add_argument("--output-dir", help="Directory where camera recordings will be stored.")
    parser.add_argument("--cameras", nargs="*", type=int, metavar="INDEX", help="Camera indices to use.")
    parser.add_argument("--auto-discover", action="store_true", help="Automatically use all detected cameras when --cameras is omitted.")
    parser.add_argument("--list", action="store_true", help="List available cameras and exit.")
    parser.add_argument("--frame-width", type=int, help="Force capture width for the cameras.")
    parser.add_argument("--frame-height", type=int, help="Force capture height for the cameras.")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording frame rate (default: 30).")
    parser.add_argument("--codec", default="MJPG", help="FourCC codec for recordings (default: MJPG).")
    parser.add_argument("--camera-fourcc", help="Attempt to configure cameras to this FOURCC (default: MJPG).")
    parser.add_argument("--preview-scale", type=float, default=1.0, help="Scale factor applied to camera preview windows when enabled.")
    parser.add_argument("--show-preview", action="store_true", help="Display camera preview windows during calibration.")
    parser.add_argument("--window-prefix", help="Custom label prefix for camera preview windows.")
    parser.add_argument("--window-width", type=int, default=1280, help="Calibration window width when not fullscreen.")
    parser.add_argument("--window-height", type=int, default=720, help="Calibration window height when not fullscreen.")
    parser.add_argument("--window-x", type=int, help="Optional X position for the calibration window.")
    parser.add_argument("--window-y", type=int, help="Optional Y position for the calibration window.")
    parser.add_argument("--point-duration", type=float, default=2.0, help="Seconds each calibration point is displayed (default: 2.0).")
    parser.add_argument("--pause-duration", type=float, default=0.6, help="Pause between points in seconds (default: 0.6).")
    parser.add_argument("--countdown-duration", type=float, default=1.5, help="Countdown before the first point (default: 1.5).")
    parser.add_argument("--target-radius", type=int, default=18, help="Radius of the calibration target in pixels (default: 18).")
    parser.add_argument("--windowed", action="store_true", help="Disable fullscreen mode for the calibration window.")
    parser.add_argument("--file-extension", help="Override output file extension (default: .avi when codec is MJPG, otherwise .mp4).")
    parser.add_argument("--stop-key", default="q", help="Keyboard key to stop recording early (default: q).")
    parser.add_argument("--align", action="store_true", help="Launch an interactive alignment UI before recording.")
    parser.add_argument("--align-snapshot", type=Path, help="Optional path for a stitched snapshot captured from the alignment UI.")
    parser.add_argument("--align-save", type=Path, help="Optional JSON file where alignment coordinates are written.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        cameras = list_available_cameras()
        if cameras:
            print("Available cameras:", " ".join(map(str, cameras)))
        else:
            print("No cameras detected.")
        if not (args.output_dir or args.cameras or args.auto_discover):
            return 0

    try:
        frame_size = _resolve_frame_size(args.frame_width, args.frame_height)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        camera_indices = _resolve_camera_indices(args.cameras or [], args.auto_discover)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    window_size = (int(args.window_width), int(args.window_height))
    window_position: Optional[Tuple[int, int]] = None
    if args.window_x is not None and args.window_y is not None:
        window_position = (args.window_x, args.window_y)
    elif (args.window_x is None) != (args.window_y is None):
        print(
            "Warning: --window-x and --window-y must be specified together; ignoring position offsets.",
            file=sys.stderr,
        )
    try:
        result = record_calibration_session(
            camera_indices,
            output_dir=args.output_dir,
            frame_size=frame_size,
            fps=args.fps,
            codec=args.codec,
            camera_fourcc=args.camera_fourcc or args.codec,
            show_preview=args.show_preview,
            window_prefix=args.window_prefix or "Camera",
            preview_scale=args.preview_scale,
            window_size=window_size,
            window_position=window_position,
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
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result.video_paths:
        print("Saved recordings:")
        for path in result.video_paths:
            print(" -", path)
    else:
        print("Calibration session complete; no recordings were written.")

    if result.events:
        print("Calibration points:")
        for event in result.events:
            print(
                " - {label} (#{idx}): {start:.3f}s -> {end:.3f}s | norm=({nx:.3f}, {ny:.3f}) | pixel={pixel}".format(
                    label=event.label,
                    idx=event.point_index + 1,
                    start=event.start_time,
                    end=event.end_time,
                    nx=event.normalized_position[0],
                    ny=event.normalized_position[1],
                    pixel=event.pixel_position,
                )
            )
    else:
        print("No calibration points recorded.")

    if result.frame_logs:
        print("Frame timestamp logs:")
        for log in result.frame_logs:
            location = str(log.csv_path) if log.csv_path is not None else "(not written)"
            frames = len(log.timestamps)
            if frames > 1:
                duration = log.timestamps[-1] - log.timestamps[0]
                observed_fps = (frames - 1) / duration if duration > 0 else None
            else:
                observed_fps = None
            fps_display = f"{observed_fps:.2f} fps" if observed_fps else "fps unknown"
            print(
                f" - camera {log.camera_index}: {frames} frames captured ({fps_display}), timestamps -> {location}"
            )

    if result.aborted:
        print("Session ended early (stop key pressed).")

    return 0


def build_calibration_points(grid_size: Tuple[int, int] = (3, 3), margin: float = 0.2) -> Tuple[CalibrationPoint, ...]:
    """Generate evenly spaced calibration points for a given grid."""
    cols, rows = grid_size
    if cols <= 0 or rows <= 0:
        raise ValueError("grid_size must contain positive values")
    if not (0 <= margin < 0.5):
        raise ValueError("margin must be in [0, 0.5)")

    if cols == 1:
        x_positions = [0.5]
    else:
        x_positions = np.linspace(margin, 1.0 - margin, cols)
    if rows == 1:
        y_positions = [0.5]
    else:
        y_positions = np.linspace(margin, 1.0 - margin, rows)

    points: List[CalibrationPoint] = []
    for row_idx, y_norm in enumerate(y_positions):
        for col_idx, x_norm in enumerate(x_positions):
            label = f"r{row_idx + 1}_c{col_idx + 1}"
            points.append(CalibrationPoint(label, (float(x_norm), float(y_norm))))
    return tuple(points)


__all__ = [
    "CalibrationPoint",
    "CalibrationSequencer",
    "CalibrationRenderer",
    "CalibrationPhaseState",
    "CalibrationEvent",
    "CalibrationRecordingResult",
    "CameraFrameLog",
    "record_calibration_session",
    "DEFAULT_CALIBRATION_POINTS",
    "build_calibration_points",
]


if __name__ == "__main__":
    sys.exit(main())
