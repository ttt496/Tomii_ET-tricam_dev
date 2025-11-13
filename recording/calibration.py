"""Calibration UI utilities for multi-camera gaze data capture."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, TextIO, Tuple

import cv2
import numpy as np
import pygame

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



# 25個の点を生成 (0.1, 0.3, 0.5, 0.7, 0.9 の組み合わせ)
_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
_all_25_points = tuple(
    CalibrationPoint(f"point_{i}_{j}", (_coords[j], _coords[i]))
    for i in range(5)
    for j in range(5)
)

# 25個の点から3セット、各9個ずつをバランス良く選択（上下左右対称性を考慮）
# 各セットが画面全体に均等に分布し、対称的な配置になるように分割
# 
# グリッド配置（i=行, j=列）:
#   (0,0) (0,1) (0,2) (0,3) (0,4)  # 上段
#   (1,0) (1,1) (1,2) (1,3) (1,4)  # 中上段
#   (2,0) (2,1) (2,2) (2,3) (2,4)  # 中央段
#   (3,0) (3,1) (3,2) (3,3) (3,4)  # 中下段
#   (4,0) (4,1) (4,2) (4,3) (4,4)  # 下段
#
# 対称性: 上下対称 (i, j) <-> (4-i, j), 左右対称 (i, j) <-> (i, 4-j)
# 各セットは上下左右対称のパターンを持ち、画面全体を均等にカバー

pointset1 = tuple(
    _all_25_points[i * 5 + j]
    for i, j in [
        (0, 0), (0, 2), (0, 4),      # 上段: 左、中央、右（左右対称）
        (1, 1), (1, 3),              # 中上段: 中左、中右（左右対称）
        (2, 0), (2, 4),              # 中央段: 左、右（左右対称）
        (3, 1), (3, 3),              # 中下段: 中左、中右（左右対称、上下対称でpointset1と対応）
        (4, 2),                      # 下段: 中央（上下対称でpointset1と対応）
    ]
)  # 9個 - 左右対称、上下対称のパターン

pointset2 = tuple(
    _all_25_points[i * 5 + j]
    for i, j in [
        (0, 1), (0, 3),              # 上段: 中左、中右（左右対称）
        (1, 0), (1, 2), (1, 4),      # 中上段: 左、中央、右（左右対称）
        (2, 1), (2, 2), (2, 3),      # 中央段: 中左、中央、中右（左右対称）
        (3, 0), (3, 2), (3, 4),      # 中下段: 左、中央、右（左右対称、上下対称でpointset2と対応）
    ]
)  # 9個 - 左右対称、上下対称のパターン

pointset3 = tuple(
    _all_25_points[i * 5 + j]
    for i, j in [
        (0, 0), (0, 4),              # 上段: 左、右（左右対称）
        (1, 1), (1, 3),              # 中上段: 中左、中右（左右対称）
        (2, 0), (2, 2), (2, 4),      # 中央段: 左、中央、右（左右対称）
        (3, 1), (3, 3),              # 中下段: 中左、中右（左右対称）
        (4, 0), (4, 4),              # 下段: 左、右（左右対称、上下対称でpointset3と対応）
    ]
)  # 9個 - 左右対称、上下対称のパターン（(0,0), (0,4), (1,1), (1,3), (2,0), (2,4), (3,1), (3,3)はpointset1と重複、合計8個の重複）

# ポイントセットの辞書
CALIBRATION_POINTSETS = {
    1: pointset1,
    2: pointset2,
    3: pointset3,
}

DEFAULT_CALIBRATION_POINTS: Tuple[CalibrationPoint, ...] = pointset1


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

    @property
    def background_color(self) -> Tuple[int, int, int]:
        return self._background_color

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

    def clone_with_window_size(self, window_size: Tuple[int, int]) -> "CalibrationRenderer":
        return CalibrationRenderer(
            window_size=window_size,
            background_color=self._background_color,
            target_color=self._target_color,
            pause_color=self._pause_color,
            text_color=self._text_color,
            radius=self._radius,
            thickness=self._thickness,
            font_scale=self._font_scale,
        )


def _bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (color[2], color[1], color[0])


def _setup_calibration_display(
    *,
    calibration_window_name: str,
    renderer: CalibrationRenderer,
    fullscreen: bool,
    window_position: Optional[Tuple[int, int]],
) -> Tuple[pygame.Surface, CalibrationRenderer]:
    if not pygame.get_init():
        pygame.init()

    if window_position is not None and not fullscreen:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{int(window_position[0])},{int(window_position[1])}"
    flags = pygame.FULLSCREEN if fullscreen else pygame.SHOWN | pygame.RESIZABLE
    screen = pygame.display.set_mode(renderer.window_size, flags)
    pygame.display.set_caption(calibration_window_name)

    actual_size = screen.get_size()
    if actual_size != renderer.window_size:
        renderer = renderer.clone_with_window_size(actual_size)

    screen.fill(_bgr_to_rgb(renderer.background_color))
    pygame.display.flip()
    return screen, renderer


def _teardown_calibration_display() -> None:
    if pygame.get_init():
        pygame.display.quit()
        pygame.quit()


def _blit_frame_to_screen(screen: pygame.Surface, frame: np.ndarray) -> None:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.image.frombuffer(rgb_frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def _poll_pygame_events(stop_key_code: Optional[int]) -> Tuple[bool, bool]:
    skip_requested = False
    abort_requested = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            abort_requested = True
        elif event.type == pygame.KEYDOWN:
            if stop_key_code is not None and event.key == stop_key_code:
                abort_requested = True
            elif event.key == pygame.K_ESCAPE:
                abort_requested = True
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                skip_requested = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            skip_requested = True
    return skip_requested, abort_requested


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
    # Reader/Writer threading structures
    write_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1024))  # バッファサイズを増やしてI/Oボトルネックを緩和
    capture_thread: Optional[threading.Thread] = None
    writer_thread: Optional[threading.Thread] = None
    # Latest frame snapshot for preview/logging (not consumed)
    latest_frame: Optional[np.ndarray] = None
    latest_timestamp: float = -1.0
    latest_lock: threading.Lock = field(default_factory=threading.Lock)
    last_logged_timestamp: float = -1.0
    # プロファイリング用統計
    read_stats: dict = field(default_factory=lambda: {"count": 0, "total_time": 0.0, "max_time": 0.0, "queue_full": 0})
    write_stats: dict = field(default_factory=lambda: {"count": 0, "total_time": 0.0, "max_time": 0.0, "queue_empty": 0})


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
    *,
    backend_first: Literal["auto", "dshow", "msmf"] = "auto",
    strict_frame_size: bool = False,
) -> cv2.VideoCapture:
    """Open camera robustly by trying multiple API backends on Windows.

    特に Windows では MSMF/DSHOW の相性で取りこぼすことがあるため、
    DirectShow → MSMF → AUTO の順で試行する。
    """
    # Decide backend probing order
    if backend_first == "msmf":
        api_preferences = [
            getattr(cv2, "CAP_MSMF", 1400),
            getattr(cv2, "CAP_DSHOW", 700),
            getattr(cv2, "CAP_ANY", 0),
        ]
    elif backend_first == "dshow":
        api_preferences = [
            getattr(cv2, "CAP_DSHOW", 700),
            getattr(cv2, "CAP_MSMF", 1400),
            getattr(cv2, "CAP_ANY", 0),
        ]
    else:
        api_preferences = [
            getattr(cv2, "CAP_DSHOW", 700),
            getattr(cv2, "CAP_MSMF", 1400),
            getattr(cv2, "CAP_ANY", 0),
        ]

    backend_names = {
        getattr(cv2, "CAP_DSHOW", 700): "DSHOW",
        getattr(cv2, "CAP_MSMF", 1400): "MSMF",
        getattr(cv2, "CAP_ANY", 0): "AUTO",
    }
    last_error = None

    for api_pref in api_preferences:
        backend_name = backend_names.get(api_pref, f"UNKNOWN({api_pref})")
        cap = cv2.VideoCapture(device_idx, api_pref)
        if not cap.isOpened():
            cap.release()
            last_error = f"{backend_name}: failed to open"
            continue

        try:
            # check_camera_fourrcc.pyと完全に同じ順序で設定（FourCC → 解像度 → FPS）
            if fourcc:
                code = fourcc.upper()
                if len(code) == 4:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))

            if frame_size is not None:
                width, height = frame_size
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

            if target_fps is not None and target_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(target_fps))
        except cv2.error:
            pass

        # check_camera_fourrcc.pyと同じように、1回だけフレームを読み取る
        success, _ = cap.read()
        used_alt_fourcc = False

        # フレーム読み取り後にBUFFERSIZEを設定（FourCC設定に影響しないように）
        if success:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
        if not success:
            cap.release()
            last_error = f"{backend_name}: failed to read frame"
            continue

        # Verify actual frame size matches request if provided
        # 注: カメラによっては要求解像度をサポートしていない場合があるため、
        # 解像度が一致しない場合でも警告を出して続行する（エラーにはしない）
        if frame_size is not None:
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            requested_width, requested_height = frame_size
            # Strict mode: try alternate FOURCCs (MJPG/YUY2) for exact size before switching backend
            if strict_frame_size and (
                abs(actual_width - requested_width) > 1 or abs(actual_height - requested_height) > 1
            ):
                for alt in ("MJPG", "YUY2"):
                    if isinstance(fourcc, str) and alt.upper() == fourcc.upper():
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*alt))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(requested_width))
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(requested_height))
                        if target_fps is not None and target_fps > 0:
                            cap.set(cv2.CAP_PROP_FPS, float(target_fps))
                        ok2, _ = cap.read()
                        if ok2:
                            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if abs(aw - requested_width) <= 1 and abs(ah - requested_height) <= 1:
                                actual_width, actual_height = aw, ah
                                used_alt_fourcc = True
                                break
                    except cv2.error:
                        pass
                if not used_alt_fourcc and (
                    abs(actual_width - requested_width) > 1 or abs(actual_height - requested_height) > 1
                ):
                    cap.release()
                    last_error = (
                        f"{backend_name}: resolution mismatch (got {actual_width}x{actual_height}, "
                        f"expected {requested_width}x{requested_height})"
                    )
                    continue
            # 解像度が一致しない場合（±1ピクセルの許容誤差）
            if abs(actual_width - requested_width) > 1 or abs(actual_height - requested_height) > 1:
                # 解像度が一致しない場合でも続行する（カメラが要求解像度をサポートしていない可能性がある）
                # 警告は後で表示されるため、ここではスキップせずに続行
                pass

        # Verify actual FOURCC matches request if provided; otherwise fallback
        if fourcc and not used_alt_fourcc:
            try:
                reported = int(cap.get(cv2.CAP_PROP_FOURCC))
                # 符号付き整数として解釈される可能性があるため、符号なし32ビット整数に変換
                actual_fourcc_int = reported & 0xFFFFFFFF
                actual_raw = "".join(chr((actual_fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
                # 制御文字や非表示文字を除去してクリーンなFourCC文字列を取得
                actual_clean = "".join(c for c in actual_raw if 32 <= ord(c) < 127).strip()
                # 非表示文字が含まれている場合でも、整数値で比較する
                # MJPG = 0x47504A4D = 'MJPG' (リトルエンディアン)
                # YUY2 = 0x32595559 = 'YUY2' (リトルエンディアン)
                expected_fourcc_int = cv2.VideoWriter_fourcc(*fourcc.upper()) & 0xFFFFFFFF
                
                # 文字列と整数値の両方で検証
                actual = actual_clean if actual_clean else None
                fourcc_matches = False
                
                if actual and actual.upper() == fourcc.upper():
                    fourcc_matches = True
                elif actual_fourcc_int == expected_fourcc_int:
                    fourcc_matches = True
                # 非表示文字が含まれている場合、整数値で比較
                elif abs(actual_fourcc_int - expected_fourcc_int) < 256:  # 小さな差は無視（エンディアンの違いなど）
                    fourcc_matches = True
                
                if not fourcc_matches:
                    # FourCCが一致しない場合
                    actual_display = actual if actual else f"0x{actual_fourcc_int:08x}"
                    # DSHOWバックエンドでは、FourCCが設定できなくても動作する可能性があるため、
                    # 警告を出して続行を許可する
                    if backend_name == "DSHOW":
                        # DSHOWではFourCCが期待通りでなくても動作する可能性があるため、警告のみ
                        warnings.warn(
                            f"Camera {device_idx} ({backend_name}): FourCC mismatch "
                            f"(got '{actual_display}'/0x{actual_fourcc_int:08x}, expected '{fourcc.upper()}'/0x{expected_fourcc_int:08x}), but continuing anyway"
                        )
                        # DSHOWで続行
                        return cap
                    else:
                        # 他のバックエンドでは厳密に検証
                        cap.release()
                        last_error = f"{backend_name}: FourCC mismatch (got '{actual_display}'/0x{actual_fourcc_int:08x}, expected '{fourcc.upper()}'/0x{expected_fourcc_int:08x})"
                        continue
            except Exception as e:
                # FourCCの取得に失敗した場合、警告を出して続行
                warnings.warn(
                    f"Camera {device_idx} ({backend_name}): Failed to verify FourCC: {e}, but continuing anyway"
                )

        return cap

    error_msg = f"Failed to open camera {device_idx} on any backend"
    if last_error:
        error_msg += f" (last attempt: {last_error})"
    raise RuntimeError(error_msg)


def _init_writer(
    output_path: Path,
    frame_size: Tuple[int, int],
    fps: float,
    codec: str,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
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


def _prepare_output_directory(output_dir: Optional[Path | str]) -> Tuple[Path, str]:
    base_dir = Path(output_dir).expanduser().resolve() if output_dir else Path("data").expanduser().resolve()
    date_component = time.strftime("%Y%m%d")
    time_component = time.strftime("%H%M%S")
    session_dir_name = f"{date_component}/{time_component}"
    output_directory = base_dir / date_component / time_component
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory, session_dir_name


def _choose_extension(codec: str, file_extension: Optional[str]) -> str:
    if file_extension:
        return file_extension if file_extension.startswith(".") else f".{file_extension}"
    return ".avi" if codec.upper() == "MJPG" else ".mp4"


def _open_sessions_for_cameras(
    camera_indexes: Sequence[int],
    *,
    frame_size: Optional[Tuple[int, int]],
    fps: float,
    camera_fourcc: Optional[str],
    codec: str,
    output_directory: Path,
    chosen_extension: str,
    show_preview: bool,
    window_prefix: str,
    preview_scale: float,
) -> Tuple[list[_CalibrationSession], list[_CalibrationSession], list[Path], list[dict]]:
    sessions: list[_CalibrationSession] = []
    all_sessions: list[_CalibrationSession] = []
    video_paths: list[Path] = []
    camera_meta: list[dict] = []

    for idx in camera_indexes:
        cap = _open_camera(
            idx,
            frame_size,
            fps,
            camera_fourcc,
            backend_first=backend_first,
            strict_frame_size=strict_frame_size,
        )
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps_val = cap.get(cv2.CAP_PROP_FPS)
        try:
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
        except Exception:
            fourcc_str = None

        output_path = output_directory / f"camera{idx}{chosen_extension}"
        writer = _init_writer(output_path, (actual_width, actual_height), fps, codec)
        video_paths.append(output_path)

        timestamp_path = output_directory / f"camera{idx}_timestamps.csv"
        timestamp_handle = timestamp_path.open("w", encoding="utf-8", newline="")
        timestamp_handle.write(
            "frame_index,timestamp_sec,phase,point_index,point_label,norm_x,norm_y,pixel_x,pixel_y\n"
        )

        window_name: Optional[str] = None
        if show_preview:
            window_name = f"{window_prefix} {idx}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if preview_scale != 1.0:
                cv2.resizeWindow(window_name, int(actual_width * preview_scale), int(actual_height * preview_scale))

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

        pixel_count = int(actual_width * actual_height)
        camera_meta.append(
            {
                "index": int(idx),
                "video_path": str(output_path),
                "timestamp_csv": str(timestamp_path),
                "actual_width": int(actual_width),
                "actual_height": int(actual_height),
                "pixel_count": pixel_count,
                "image_size": {"width": int(actual_width), "height": int(actual_height)},
                "requested_fps": float(fps),
                "driver_reported_fps": float(actual_fps_val) if actual_fps_val and actual_fps_val > 0 else None,
                "fourcc": fourcc_str,
            }
        )

    return sessions, all_sessions, video_paths, camera_meta


def _write_session_meta(
    output_directory: Path,
    session_dir_name: str,
    *,
    camera_meta: list[dict],
    renderer: "CalibrationRenderer",
    fullscreen: bool,
    window_position: Optional[Tuple[int, int]],
    point_sequence: Sequence["CalibrationPoint"],
    sequencer: "CalibrationSequencer",
    target_radius: int,
    codec: str,
    camera_fourcc: Optional[str],
    chosen_extension: str,
) -> None:
    try:
        meta = {
            "session_dir": str(output_directory),
            "timestamp": session_dir_name,
            "cameras": camera_meta,
            "window": {
                "fullscreen": bool(fullscreen),
                "size": {"width": int(renderer.window_size[0]), "height": int(renderer.window_size[1])},
                "position": {"x": int(window_position[0]), "y": int(window_position[1])} if window_position is not None else None,
            },
            "calibration": {
                "points": [
                    {
                        "label": p.label,
                        "normalized": {"x": float(p.normalized_position[0]), "y": float(p.normalized_position[1])},
                    }
                    for p in point_sequence
                ],
                "durations": {
                    "point": float(sequencer.point_duration),
                    "pause": float(sequencer.pause_duration),
                    "countdown": float(sequencer.countdown_duration),
                    "total": float(sequencer.total_duration),
                },
                "target_radius": int(target_radius),
            },
            "encoding": {
                "codec": str(codec),
                "camera_fourcc": str(camera_fourcc),
                "file_extension": str(chosen_extension),
            },
        }
        (output_directory / "session_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _camera_capture_thread(session: _CalibrationSession, stop_event: threading.Event) -> None:
    """カメラからフレームを読み取り続けるスレッド関数"""
    read_fail_count = 0
    # 複数カメラ時のUSB帯域競合を緩和するため、各カメラスレッドに少しオフセットを追加
    time.sleep(session.index * 0.05)  # カメラごとに50msずつオフセット（USB帯域競合を大幅に緩和）
    
    while not stop_event.is_set() and session.active:
        try:
            # カメラが開いているか確認
            if not session.capture.isOpened():
                print(f"Error: Camera {session.index} is not opened", file=sys.stderr)
                session.active = False
                break
            
            read_start = time.monotonic()
            # 通常のread()を使用（grab/retrieveの分離は一部のカメラで不安定なため）
            # USB帯域競合によるブロッキングは、バッファサイズを1に設定することで緩和
            try:
                success, frame = session.capture.read()
            except Exception as read_error:
                print(f"Error: Camera {session.index} read() failed: {read_error}", file=sys.stderr)
                read_fail_count += 1
                if read_fail_count > 10:
                    session.active = False
                    break
                time.sleep(0.01)  # 失敗時は少し待つ
                continue
            
            read_time = time.monotonic() - read_start
            
            if not success:
                read_fail_count += 1
                if read_fail_count > 10:
                    session.active = False
                    break
                time.sleep(0.01)  # 失敗時は少し待つ
                continue
            
            read_fail_count = 0
            timestamp = time.monotonic()
            
            # 統計を更新（各スレッドが自分のセッションの統計のみを更新するため、ロック不要）
            session.read_stats["count"] += 1
            session.read_stats["total_time"] += read_time
            # max_timeの更新は競合する可能性があるが、統計用途なので問題ない
            if read_time > session.read_stats["max_time"]:
                session.read_stats["max_time"] = read_time
            
            # 読み取り時間が異常に長い場合は警告
            if read_time > 0.1:
                print(f"Warning: Camera {session.index} read took {read_time*1000:.1f}ms", file=sys.stderr)
            
            # 書き込みキューへ投入（I/Oボトルネックを緩和するため、バッファサイズを大きくしている）
            # フレーム読み取りを優先するため、書き込みキューへの投入を先に行う
            queue_put_success = False
            try:
                session.write_queue.put((timestamp, frame), block=False)
                queue_put_success = True
            except queue.Full:
                session.read_stats["queue_full"] += 1
                # キューが満杯の場合、短いタイムアウトで再試行
                try:
                    session.write_queue.put((timestamp, frame), timeout=0.001)  # タイムアウトを短縮
                    queue_put_success = True
                except queue.Full:
                    # それでも満杯の場合は、古いフレームを1つ捨てて新しいフレームを入れる
                    try:
                        session.write_queue.get_nowait()
                        session.write_queue.put((timestamp, frame), block=False)
                        queue_put_success = True
                    except queue.Empty:
                        pass
            
            # 最新フレームを更新（プレビュー/ログ用）
            # フレーム読み取りを優先するため、コピーは最小限に（必要時のみ）
            # 書き込みキューへの投入が成功した場合のみ、プレビュー用のコピーを作成
            if queue_put_success:
                # ロック時間を最小限にするため、フレームのコピーはロック外で行う
                frame_copy = frame.copy() if frame is not None else None
                with session.latest_lock:
                    session.latest_timestamp = timestamp
                    session.latest_frame = frame_copy
            else:
                # キューが満杯で書き込めない場合でも、タイムスタンプだけ更新（フレームは更新しない）
                with session.latest_lock:
                    session.latest_timestamp = timestamp
                    # フレームは更新しない（古いフレームを保持）
        except Exception as e:
            print(f"Error in camera {session.index} thread: {e}", file=sys.stderr)
            session.active = False
            break


def _start_camera_threads(sessions: list[_CalibrationSession]) -> threading.Event:
    """全カメラのキャプチャスレッドを開始"""
    stop_event = threading.Event()
    for session in sessions:
        # キャプチャスレッド：高優先度でフレーム読み取りを優先
        thread = threading.Thread(
            target=_camera_capture_thread,
            args=(session, stop_event),
            daemon=True,
            name=f"CameraCapture-{session.index}",
        )
        session.capture_thread = thread
        thread.start()
        # writer thread
        def _writer_loop(sess: _CalibrationSession, stop_evt: threading.Event) -> None:
            """書き込みスレッド：キューからフレームを取得してディスクに書き込む"""
            # バッチ書き込み用のバッファ（I/O効率を向上、将来的に拡張可能）
            batch_buffer = []
            batch_size = 1  # 現在は1フレームずつ（OpenCVのVideoWriterの制約）
            
            while not stop_evt.is_set() and sess.active:
                try:
                    # タイムアウトを短くして、キューが空でもすぐに次のループに進む
                    _, frm = sess.write_queue.get(timeout=0.05)
                    batch_buffer.append(frm)
                except queue.Empty:
                    sess.write_stats["queue_empty"] += 1
                    # バッファにフレームが残っている場合は書き込む
                    if batch_buffer and sess.writer is not None:
                        write_start = time.monotonic()
                        for buffered_frame in batch_buffer:
                            sess.writer.write(buffered_frame)
                        write_time = time.monotonic() - write_start
                        sess.write_stats["count"] += len(batch_buffer)
                        sess.write_stats["total_time"] += write_time
                        if write_time > sess.write_stats["max_time"]:
                            sess.write_stats["max_time"] = write_time
                        batch_buffer.clear()
                    continue
                
                # バッファが一定サイズに達したら書き込む（現在は1フレームずつ）
                if len(batch_buffer) >= batch_size and sess.writer is not None:
                    write_start = time.monotonic()
                    for buffered_frame in batch_buffer:
                        sess.writer.write(buffered_frame)
                    write_time = time.monotonic() - write_start
                    sess.write_stats["count"] += len(batch_buffer)
                    sess.write_stats["total_time"] += write_time
                    if write_time > sess.write_stats["max_time"]:
                        sess.write_stats["max_time"] = write_time
                    # 書き込み時間が異常に長い場合は警告（ディスクI/Oのボトルネックの可能性）
                    if write_time > 0.1:
                        print(
                            f"Warning: Camera {sess.index} write took {write_time*1000:.1f}ms "
                            f"({len(batch_buffer)} frames, possible disk I/O bottleneck)",
                            file=sys.stderr,
                        )
                    batch_buffer.clear()
            
            # 終了時にバッファに残っているフレームを書き込む
            if batch_buffer and sess.writer is not None:
                for buffered_frame in batch_buffer:
                    sess.writer.write(buffered_frame)
                sess.write_stats["count"] += len(batch_buffer)
        # 書き込みスレッド：通常優先度（I/O待機が多いため）
        wthread = threading.Thread(
            target=_writer_loop,
            args=(session, stop_event),
            daemon=True,
            name=f"CameraWriter-{session.index}",
        )
        session.writer_thread = wthread
        wthread.start()
    return stop_event


def _stop_camera_threads(sessions: list[_CalibrationSession], stop_event: threading.Event) -> None:
    """全カメラのキャプチャスレッドを停止"""
    stop_event.set()
    for session in sessions:
        if session.capture_thread is not None:
            session.capture_thread.join(timeout=1.0)
        if session.writer_thread is not None:
            session.writer_thread.join(timeout=1.0)


def _run_calibration_loop(
    *,
    sessions: list[_CalibrationSession],
    all_sessions: list[_CalibrationSession],
    renderer: "CalibrationRenderer",
    sequencer: "CalibrationSequencer",
    screen: pygame.Surface,
    stop_key: str,
    show_preview: bool,
) -> Tuple[bool, list[_PointRecord], bool]:
    frames_captured = False
    aborted = False
    point_records: list[_PointRecord] = []
    active_point_index: Optional[int] = None
    skip_trigger = {"flag": False}
    manual_offset = 0.0
    clock = pygame.time.Clock()
    stop_key_code: Optional[int] = None
    for candidate in (stop_key, stop_key.lower(), stop_key.upper()):
        try:
            stop_key_code = pygame.key.key_code(candidate)
            break
        except (ValueError, AttributeError):
            continue

    start_time = time.monotonic()
    
    # カメラキャプチャスレッドを開始
    stop_event = _start_camera_threads(sessions)
    
    # プロファイリング用の変数
    profile_stats = {
        "loop_iterations": 0,
        "frames_processed": 0,
        "queue_empty_count": 0,
        "frame_write_time": [],
        "render_time": [],
        "total_loop_time": [],
    }
    last_profile_time = start_time
    profile_interval = 3.0  # 2秒ごとに統計を表示

    try:
        while True:
            loop_start = time.monotonic()
            now = time.monotonic()
            elapsed = now - start_time
            adjusted_elapsed = elapsed + manual_offset
            state = sequencer.state_for_elapsed(adjusted_elapsed)
            stage_start_real = max(0.0, elapsed - state.elapsed)
            profile_stats["loop_iterations"] += 1

            skip_from_input, abort_from_input = _poll_pygame_events(stop_key_code)
            if abort_from_input:
                aborted = True
                break
            if skip_from_input:
                skip_trigger["flag"] = True

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
                    norm_pos = state.point.normalized_position
                    pixel_pos = renderer.normalized_to_pixel(norm_pos)
                    point_records.append(
                        _PointRecord(
                            point_index=state.point_index,
                            label=state.point.label,
                            start_time=stage_start_real,
                            normalized_position=norm_pos,
                            pixel_position=pixel_pos,
                        )
                    )
            else:
                if active_point_index is not None and point_records:
                    last = point_records[-1]
                    if last.end_time is None:
                        last.end_time = stage_start_real
                active_point_index = None

            point_idx_for_log: Optional[int] = None
            point_label_for_log: Optional[str] = None
            norm_for_log: Optional[Tuple[float, float]] = None
            pixel_for_log: Optional[Tuple[int, int]] = None
            if state.phase == "point" and state.point is not None:
                point_idx_for_log = state.point_index
                point_label_for_log = state.point.label
                norm_for_log = tuple(float(v) for v in state.point.normalized_position)
                pixel_for_log = renderer.normalized_to_pixel(state.point.normalized_position)

            for session in list(sessions):
                # カメラが停止した場合はセッションを削除
                if not session.active:
                    print(
                        f"Warning: Camera {session.index} stopped delivering frames; closing stream.",
                        file=sys.stderr,
                    )
                    _close_session(session)
                    sessions.remove(session)
                    continue

                # 最新フレームのスナップショットを取得（ロック時間を最小限に）
                # ロック内では参照のみを取得し、コピーはロック外で行う
                with session.latest_lock:
                    frame_timestamp = session.latest_timestamp
                    frame_ref = session.latest_frame
                # ロック外でフレームをコピー（ロック時間を最小限に）
                frame = frame_ref.copy() if frame_ref is not None else None

                if frame is None or frame_timestamp <= 0:
                    profile_stats["queue_empty_count"] += 1
                    continue

                # 同一フレームの重複ログを防ぐ（ロック外でチェック）
                if frame_timestamp <= session.last_logged_timestamp:
                    continue
                session.last_logged_timestamp = frame_timestamp

                frames_captured = True
                profile_stats["frames_processed"] += 1
                ts = frame_timestamp - start_time
                # リストへの追加は各スレッドが自分のセッションのみを操作するため、ロック不要
                session.frame_times.append(ts)
                session.frame_point_indices.append(point_idx_for_log)
                session.frame_point_labels.append(point_label_for_log)
                session.frame_point_norms.append(norm_for_log)
                session.frame_point_pixels.append(pixel_for_log)

                if session.timestamp_handle is not None:
                    idx_value = "" if point_idx_for_log is None else str(point_idx_for_log)
                    label_value = "" if point_label_for_log is None else point_label_for_log
                    if norm_for_log is not None:
                        nx, ny = norm_for_log
                        nx_s, ny_s = f"{nx:.6f}", f"{ny:.6f}"
                        if pixel_for_log is not None:
                            px, py = pixel_for_log
                            px_s, py_s = str(int(px)), str(int(py))
                        else:
                            px_s = py_s = ""
                    else:
                        nx_s = ny_s = px_s = py_s = ""
                    session.timestamp_handle.write(
                        f"{session.frame_index},{ts:.6f},{state.phase},{idx_value},{label_value},{nx_s},{ny_s},{px_s},{py_s}\n"
                    )
                session.frame_index += 1

                # 書き込みは writer スレッドに任せる

                if show_preview and session.window_name is not None:
                    cv2.imshow(session.window_name, frame)

            if not sessions:
                raise RuntimeError("All cameras stopped delivering frames.")

            render_start = time.monotonic()
            frame_ui = renderer.render(state)
            _blit_frame_to_screen(screen, frame_ui)
            profile_stats["render_time"].append(time.monotonic() - render_start)
            
            profile_stats["total_loop_time"].append(time.monotonic() - loop_start)
            clock.tick(240)
            
            # 定期的に統計情報を表示
            current_time = time.monotonic()
            if current_time - last_profile_time >= profile_interval:
                elapsed_profiling = current_time - last_profile_time
                loop_rate = profile_stats["loop_iterations"] / elapsed_profiling
                frame_rate = profile_stats["frames_processed"] / elapsed_profiling
                queue_empty_rate = profile_stats["queue_empty_count"] / profile_stats["loop_iterations"] * 100 if profile_stats["loop_iterations"] > 0 else 0
                
                print(f"\n[プロファイリング統計] (過去{elapsed_profiling:.1f}秒)", file=sys.stderr)
                print(f"  ループ回数: {profile_stats['loop_iterations']} ({loop_rate:.1f} loops/sec)", file=sys.stderr)
                print(f"  フレーム処理数: {profile_stats['frames_processed']} ({frame_rate:.2f} fps)", file=sys.stderr)
                print(f"  キュー空の回数: {profile_stats['queue_empty_count']} ({queue_empty_rate:.1f}%)", file=sys.stderr)
                
                # 各カメラごとの詳細統計
                for session in all_sessions:
                    if session.read_stats["count"] > 0:
                        read_count = session.read_stats["count"]
                        read_avg = (session.read_stats["total_time"] / read_count) * 1000
                        read_max = session.read_stats["max_time"] * 1000
                        read_fps = read_count / elapsed_profiling
                        queue_full = session.read_stats["queue_full"]
                        queue_full_rate = (queue_full / read_count * 100) if read_count > 0 else 0
                        
                        write_count = session.write_stats["count"]
                        write_avg = (session.write_stats["total_time"] / write_count * 1000) if write_count > 0 else 0
                        write_max = session.write_stats["max_time"] * 1000
                        write_queue_empty = session.write_stats["queue_empty"]
                        
                        write_fps = write_count / elapsed_profiling if write_count > 0 else 0.0
                        print(f"  Camera {session.index}:", file=sys.stderr)
                        print(f"    読み取り: {read_count}回 ({read_fps:.2f} fps), avg={read_avg:.2f}ms, max={read_max:.2f}ms, キュー満杯={queue_full}回 ({queue_full_rate:.1f}%)", file=sys.stderr)
                        print(f"    書き込み: {write_count}回 ({write_fps:.2f} fps), avg={write_avg:.2f}ms, max={write_max:.2f}ms, キュー空={write_queue_empty}回", file=sys.stderr)
                        # 読み取りFPSと書き込みFPSの差が大きい場合は警告
                        if read_fps > 0 and write_fps > 0:
                            fps_diff = abs(read_fps - write_fps)
                            if fps_diff > read_fps * 0.2:  # 20%以上の差がある場合
                                print(
                                    f"    ⚠️  FPS差が大きい: 読み取り={read_fps:.2f} fps, 書き込み={write_fps:.2f} fps "
                                    f"(差={fps_diff:.2f} fps) - ディスクI/Oまたはキューがボトルネックの可能性",
                                    file=sys.stderr,
                                )
                
                if profile_stats["frame_write_time"]:
                    avg_write = sum(profile_stats["frame_write_time"]) / len(profile_stats["frame_write_time"]) * 1000
                    max_write = max(profile_stats["frame_write_time"]) * 1000
                    print(f"  フレーム書き込み: avg={avg_write:.2f}ms, max={max_write:.2f}ms", file=sys.stderr)
                
                if profile_stats["render_time"]:
                    avg_render = sum(profile_stats["render_time"]) / len(profile_stats["render_time"]) * 1000
                    max_render = max(profile_stats["render_time"]) * 1000
                    print(f"  レンダリング: avg={avg_render:.2f}ms, max={max_render:.2f}ms", file=sys.stderr)
                
                if profile_stats["total_loop_time"]:
                    avg_loop = sum(profile_stats["total_loop_time"]) / len(profile_stats["total_loop_time"]) * 1000
                    max_loop = max(profile_stats["total_loop_time"]) * 1000
                    print(f"  ループ全体: avg={avg_loop:.2f}ms, max={max_loop:.2f}ms", file=sys.stderr)
                
                # 統計をリセット
                for key in profile_stats:
                    if isinstance(profile_stats[key], list):
                        profile_stats[key] = []
                    else:
                        profile_stats[key] = 0
                # カメラ統計もリセット
                for session in all_sessions:
                    session.read_stats = {"count": 0, "total_time": 0.0, "max_time": 0.0, "queue_full": 0}
                    session.write_stats = {"count": 0, "total_time": 0.0, "max_time": 0.0, "queue_empty": 0}
                last_profile_time = current_time

            if state.phase == "finished":
                break

    finally:
        _stop_camera_threads(sessions, stop_event)

    final_elapsed = time.monotonic() - start_time
    if point_records and point_records[-1].end_time is None:
        point_records[-1].end_time = final_elapsed
    return frames_captured, point_records, aborted


def _finalize_and_collect_results(
    *,
    all_sessions: list[_CalibrationSession],
    video_paths: list[Path],
    point_records: list[_PointRecord],
    frames_captured: bool,
    aborted: bool,
    start_time: float,
) -> "CalibrationRecordingResult":
    for session in all_sessions:
        _close_session(session)
    _teardown_calibration_display()

    if not frames_captured:
        raise RuntimeError("No frames captured during calibration session.")

    final_elapsed = time.monotonic() - start_time
    events: list[CalibrationEvent] = []
    for rec in point_records:
        st = float(rec.start_time)
        et = float(rec.end_time) if rec.end_time is not None else final_elapsed
        events.append(
            CalibrationEvent(
                point_index=int(rec.point_index),
                label=str(rec.label),
                start_time=st,
                end_time=et,
                normalized_position=tuple(float(v) for v in rec.normalized_position),
                pixel_position=tuple(int(v) for v in rec.pixel_position),
            )
        )

    frame_logs: list[CameraFrameLog] = []
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

def record_calibration_session(
    camera_indexes: Sequence[int],
    output_dir: Optional[Path | str] = None,
    *,
    frame_size: Optional[Tuple[int, int]] = None,
    fps: float = 30.0,
    codec: str = "MJPG",
    camera_fourcc: Optional[str] = "MJPG",
    backend_first: Literal["auto", "dshow", "msmf"] = "auto",
    strict_frame_size: bool = False,
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
    if fullscreen and window_position is None:
        window_position = (0, 0)

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

    pygame_screen: Optional[pygame.Surface] = None
    try:
        pygame_screen, renderer = _setup_calibration_display(
            calibration_window_name=calibration_window_name,
            renderer=renderer,
            fullscreen=fullscreen,
            window_position=window_position,
        )
    except Exception:
        _teardown_calibration_display()
        raise

    # Prepare default output directory and session timestamp tag (YYYYMMDD/HHMMSS)
    base_dir = Path(output_dir).expanduser().resolve() if output_dir else Path("data").expanduser().resolve()
    date_component = time.strftime("%Y%m%d")
    time_component = time.strftime("%H%M%S")
    session_dir_name = f"{date_component}/{time_component}"
    output_directory = base_dir / date_component / time_component
    output_directory.mkdir(parents=True, exist_ok=True)

    sessions: List[_CalibrationSession] = []
    all_sessions: List[_CalibrationSession] = []
    video_paths: List[Path] = []
    camera_meta: List[dict] = []

    try:
        # すべてのカメラで統一されたFPSを使用するため、まずすべてのカメラを開いてFPSを確認
        caps_info = []
        for idx in camera_indexes:
            cap = _open_camera(
                idx,
                frame_size,
                fps,
                camera_fourcc,
                backend_first=backend_first,
                strict_frame_size=strict_frame_size,
            )
            actual_fps_val = cap.get(cv2.CAP_PROP_FPS)
            caps_info.append((idx, cap, actual_fps_val))
        
        # カメラが報告するFPSを表示（デバッグ用）
        reported_fps_list = []
        for idx, _, actual_fps_val in caps_info:
            if actual_fps_val and actual_fps_val > 0:
                reported_fps_list.append((idx, float(actual_fps_val)))
        
        if reported_fps_list:
            fps_info = ", ".join([f"Camera {idx}: {fps_val:.1f}" for idx, fps_val in reported_fps_list])
            print(f"Camera reported FPS: {fps_info}", file=sys.stderr)
        
        # すべてのカメラで統一されたwriter_fpsを決定
        # カメラが報告するFPSが要求FPSと大きく異なる場合（±10%以上）は、要求FPSを優先
        valid_fps_values = []
        for idx, _, actual_fps_val in caps_info:
            if actual_fps_val and actual_fps_val > 0.1 and actual_fps_val < 240:
                # 要求FPSの90%〜110%の範囲内かチェック（より厳密に）
                if fps * 0.9 <= actual_fps_val <= fps * 1.1:
                    valid_fps_values.append(float(actual_fps_val))
        
        # すべてのカメラが要求FPSに近い値（±10%以内）を報告している場合のみ、その平均を使用
        # それ以外は要求FPSを強制的に使用（カメラの報告が不正確な場合が多いため）
        if len(valid_fps_values) == len(caps_info) and valid_fps_values:
            # すべてのカメラが要求FPSに近い値を報告している場合
            avg_reported = sum(valid_fps_values) / len(valid_fps_values)
            # 要求FPSとの差が5%以内の場合のみ、報告された平均を使用
            if abs(avg_reported - fps) / fps <= 0.05:
                writer_fps = avg_reported
                print(
                    f"Using average of camera-reported FPS values ({writer_fps:.2f} fps) for all cameras.",
                    file=sys.stderr,
                )
            else:
                writer_fps = float(fps)
                print(
                    f"Camera-reported FPS ({avg_reported:.2f}) differs from requested ({fps:.2f}). "
                    f"Using requested FPS ({writer_fps:.2f} fps) for all cameras.",
                    file=sys.stderr,
                )
        else:
            # 一部またはすべてのカメラが要求FPSと大きく異なる値を報告している場合
            writer_fps = float(fps)
            if len(valid_fps_values) < len(caps_info):
                invalid_count = len(caps_info) - len(valid_fps_values)
                print(
                    f"Warning: {invalid_count} camera(s) reported FPS values significantly different from requested ({fps:.2f} fps). "
                    f"Using requested FPS ({writer_fps:.2f} fps) for all cameras.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Using requested FPS ({writer_fps:.2f} fps) for all cameras.",
                    file=sys.stderr,
                )
        
        # すべてのカメラで同じwriter_fpsを使用
        # 解像度の検証と表示
        if frame_size is not None:
            requested_width, requested_height = frame_size
            resolution_mismatches = []
            for idx, cap, _ in caps_info:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if abs(actual_width - requested_width) > 1 or abs(actual_height - requested_height) > 1:
                    resolution_mismatches.append((idx, actual_width, actual_height))
            
            if resolution_mismatches:
                mismatch_info = ", ".join([
                    f"Camera {idx}: {w}x{h}" for idx, w, h in resolution_mismatches
                ])
                print(
                    f"Warning: Some cameras did not set the requested resolution ({requested_width}x{requested_height}): {mismatch_info}",
                    file=sys.stderr,
                )
        
        for idx, cap, actual_fps_val in caps_info:
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            try:
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
            except Exception:
                fourcc_str = None
            writer: Optional[cv2.VideoWriter] = None
            output_path: Optional[Path] = None
            timestamp_path: Optional[Path] = None
            timestamp_handle: Optional[TextIO] = None
            # Always write outputs into data/YYMMDD-HHMMSS, name camera{n}.*
            output_path = output_directory / f"camera{idx}{chosen_extension}"
            # すべてのカメラで統一されたwriter_fpsを使用
            writer = _init_writer(output_path, (actual_width, actual_height), writer_fps, codec)
            video_paths.append(output_path)

            timestamp_path = output_directory / f"camera{idx}_timestamps.csv"
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

            pixel_count = int(actual_width * actual_height)
            camera_meta.append(
                {
                    "index": int(idx),
                    "video_path": str(output_path),
                    "timestamp_csv": str(timestamp_path),
                    "actual_width": int(actual_width),
                    "actual_height": int(actual_height),
                    "pixel_count": pixel_count,
                    "image_size": {"width": int(actual_width), "height": int(actual_height)},
                    "requested_fps": float(fps),
                    "driver_reported_fps": float(actual_fps_val) if actual_fps_val and actual_fps_val > 0 else None,
                    "writer_fps": float(writer_fps),
                    "fourcc": fourcc_str,
                }
            )

    except Exception:
        # エラーが発生した場合、開いたカメラをすべてクローズ
        for session in all_sessions:
            _close_session(session)
        # caps_infoに保存されたカメラもクローズ（all_sessionsに追加される前にエラーが発生した場合）
        if 'caps_info' in locals():
            for _, cap, _ in caps_info:
                if cap.isOpened():
                    cap.release()
        _teardown_calibration_display()
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

    # Write session-level metadata for reproducibility
    try:
        meta = {
            "session_dir": str(output_directory),
            "timestamp": session_dir_name,
            "cameras": camera_meta,
            "window": {
                "fullscreen": bool(fullscreen),
                "size": {"width": int(renderer.window_size[0]), "height": int(renderer.window_size[1])},
                "position": {"x": int(window_position[0]), "y": int(window_position[1])} if window_position is not None else None,
            },
            "calibration": {
                "points": [
                    {
                        "label": p.label,
                        "normalized": {"x": float(p.normalized_position[0]), "y": float(p.normalized_position[1])},
                    }
                    for p in point_sequence
                ],
                "durations": {
                    "point": float(point_duration),
                    "pause": float(pause_duration),
                    "countdown": float(countdown_duration),
                    "total": float(sequencer.total_duration),
                },
                "target_radius": int(target_radius),
            },
            "encoding": {
                "codec": str(codec),
                "camera_fourcc": str(camera_fourcc),
                "file_extension": str(chosen_extension),
            },
        }
        (output_directory / "session_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    start_time = time.monotonic()
    try:
        frames_captured, point_records, aborted = _run_calibration_loop(
            sessions=sessions,
            all_sessions=all_sessions,
            renderer=renderer,
            sequencer=sequencer,
            screen=pygame_screen,
            stop_key=stop_key,
            show_preview=show_preview,
        )
    except KeyboardInterrupt:
        aborted = True
        frames_captured = False
        point_records = []
    finally:
        _teardown_calibration_display()

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


def _resolve_frame_size(
    width: Optional[int],
    height: Optional[int],
    *,
    default: Optional[Tuple[int, int]] = None,
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
    parser.add_argument(
        "--frame-size",
        type=str,
        metavar="WIDTHxHEIGHT",
        help="Force capture size for the cameras (e.g., 1920x1080 or 1280x720). Overrides --frame-width and --frame-height.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=1280,
        help="Force capture width for the cameras (default fallback: 1280 unless --native-resolution). Must be used with --frame-height.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=720,
        help="Force capture height for the cameras (default fallback: 720 unless --native-resolution). Must be used with --frame-width.",
    )
    parser.add_argument(
        "--native-resolution",
        action="store_true",
        help="Use each camera's native resolution instead of forcing the 1280x720 fallback.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Recording frame rate (default: 30).")
    parser.add_argument("--codec", default="MJPG", help="FourCC codec for recordings (default: MJPG).")
    parser.add_argument("--camera-fourcc", default="MJPG", help="Attempt to configure cameras to this FOURCC (default: MJPG).")
    parser.add_argument(
        "--backend-first",
        choices=["auto", "dshow", "msmf"],
        default="auto",
        help="Which backend to try first when opening cameras (Windows).",
    )
    parser.add_argument(
        "--strict-frame-size",
        action="store_true",
        help="If set, require the requested frame size; try next backend when mismatch.",
    )
    parser.add_argument("--preview-scale", type=float, default=1.0, help="Scale factor applied to camera preview windows when enabled.")
    parser.add_argument("--show-preview", action="store_true", help="Display camera preview windows during calibration.")
    parser.add_argument("--window-prefix", help="Custom label prefix for camera preview windows.")
    parser.add_argument("--window-width", type=int, default=1920, help="Calibration window width when not fullscreen.")
    parser.add_argument("--window-height", type=int, default=1080, help="Calibration window height when not fullscreen.")
    parser.add_argument("--window-x", type=int, help="Optional X position for the calibration window.")
    parser.add_argument("--window-y", type=int, help="Optional Y position for the calibration window.")
    parser.add_argument("--point-duration", type=float, default=2.0, help="Seconds each calibration point is displayed (default: 2.0).")
    parser.add_argument("--pause-duration", type=float, default=0.6, help="Pause between points in seconds (default: 0.6).")
    parser.add_argument("--countdown-duration", type=float, default=1.5, help="Countdown before the first point (default: 1.5).")
    parser.add_argument("--target-radius", type=int, default=20, help="Radius of the calibration target in pixels (default: 18).")
    parser.add_argument("--windowed", action="store_true", help="Disable fullscreen mode for the calibration window.")
    parser.add_argument("--file-extension", help="Override output file extension (default: .avi when codec is MJPG, otherwise .mp4).")
    parser.add_argument("--stop-key", default="q", help="Keyboard key to stop recording early (default: q).")
    parser.add_argument("--align", action="store_true", help="Launch an interactive alignment UI before recording.")
    parser.add_argument("--align-snapshot", type=Path, help="Optional path for a stitched snapshot captured from the alignment UI.")
    parser.add_argument("--align-save", type=Path, help="Optional JSON file where alignment coordinates are written.")
    parser.add_argument(
        "--pointset",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Calibration point set to use (1, 2, or 3). Each set contains 9 points from a 5x5 grid (0.1, 0.3, 0.5, 0.7, 0.9). Default: 1.",
    )
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

    default_frame_size: Optional[Tuple[int, int]]
    if args.native_resolution:
        default_frame_size = None
    else:
        # FPS安定化のため、デフォルト解像度を1280x720に設定
        # より高い解像度が必要な場合は --frame-size または --frame-width/--frame-height で指定可能
        default_frame_size = (1280, 720)

    # --frame-size オプションを処理（--frame-width/--frame-height より優先）
    frame_width = args.frame_width
    frame_height = args.frame_height
    if args.frame_size:
        # --frame-size と --frame-width/--frame-height の競合チェック
        if args.frame_width is not None or args.frame_height is not None:
            print(
                "Warning: --frame-size is specified, ignoring --frame-width and --frame-height.",
                file=sys.stderr,
            )
        try:
            parts = args.frame_size.split("x")
            if len(parts) != 2:
                raise ValueError("--frame-size must be in format WIDTHxHEIGHT (e.g., 1920x1080)")
            frame_width = int(parts[0])
            frame_height = int(parts[1])
            if frame_width <= 0 or frame_height <= 0:
                raise ValueError("Frame dimensions must be positive")
        except ValueError as e:
            print(f"Error: Invalid --frame-size format: {e}", file=sys.stderr)
            return 2

    try:
        frame_size = _resolve_frame_size(
            frame_width,
            frame_height,
            default=default_frame_size,
        )
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
    # 選択されたポイントセットを取得
    selected_pointset = CALIBRATION_POINTSETS[args.pointset]
    
    try:
        result = record_calibration_session(
            camera_indices,
            output_dir=args.output_dir,
            frame_size=frame_size,
            fps=args.fps,
            codec=args.codec,
            camera_fourcc=args.camera_fourcc or args.codec,
            backend_first=args.backend_first,
            strict_frame_size=args.strict_frame_size,
            show_preview=args.show_preview,
            window_prefix=args.window_prefix or "Camera",
            preview_scale=args.preview_scale,
            window_size=window_size,
            window_position=window_position,
            points=selected_pointset,
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
        
        # カメラ情報（画素数と画像サイズ）を表示
        try:
            session_meta_path = result.video_paths[0].parent / "session_meta.json"
            if session_meta_path.exists():
                meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
                if "cameras" in meta:
                    print("\nCamera information:")
                    for cam in meta["cameras"]:
                        idx = cam.get("index", "?")
                        width = cam.get("actual_width", "?")
                        height = cam.get("actual_height", "?")
                        pixel_count = cam.get("pixel_count", width * height if isinstance(width, int) and isinstance(height, int) else "?")
                        print(f" - Camera {idx}: {width}x{height} ({pixel_count:,} pixels)")
        except Exception:
            pass
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
                # フレーム間隔の平均を計算してFPSを求める（より正確）
                intervals = []
                for i in range(1, len(log.timestamps)):
                    interval = log.timestamps[i] - log.timestamps[i - 1]
                    if interval > 0:  # 0以下の間隔は無視
                        intervals.append(interval)
                
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    observed_fps = 1.0 / avg_interval if avg_interval > 0 else None
                else:
                    # 間隔が計算できない場合は総時間から計算
                    total_duration = log.timestamps[-1] - log.timestamps[0]
                    observed_fps = (frames - 1) / total_duration if total_duration > 0 else None
            elif frames == 1:
                # フレームが1つだけの場合はFPS計算不可
                observed_fps = None
            else:
                observed_fps = None
            
            if observed_fps is not None:
                # 総時間も表示
                total_duration = log.timestamps[-1] - log.timestamps[0] if frames > 1 else 0.0
                fps_display = f"{observed_fps:.2f} fps (total: {total_duration:.2f}s)"
            else:
                fps_display = "fps unknown"
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
