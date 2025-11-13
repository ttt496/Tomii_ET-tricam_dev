"""Single-window alignment helper for positioning cameras around a display."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _Marker:
    identifier: str
    label: str
    color: Tuple[int, int, int]
    norm_x: float
    norm_y: float
    is_selected: bool = False

    @property
    def position(self) -> Tuple[float, float]:
        return self.norm_x, self.norm_y

    def set_position(self, x: float, y: float) -> None:
        self.norm_x = float(np.clip(x, 0.0, 1.0))
        self.norm_y = float(np.clip(y, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Alignment UI implementation
# ---------------------------------------------------------------------------


def _get_screen_size() -> Tuple[int, int]:
    try:
        app_services_path = ctypes.util.find_library("ApplicationServices")
        if app_services_path:
            app_services = ctypes.cdll.LoadLibrary(app_services_path)
            app_services.CGMainDisplayID.restype = ctypes.c_uint32
            display_id = app_services.CGMainDisplayID()
            app_services.CGDisplayPixelsWide.restype = ctypes.c_size_t
            app_services.CGDisplayPixelsHigh.restype = ctypes.c_size_t
            width = app_services.CGDisplayPixelsWide(display_id)
            height = app_services.CGDisplayPixelsHigh(display_id)
            if width and height:
                return int(width), int(height)
    except Exception:
        pass
    return 1920, 1080


class _AlignmentCanvas:
    def __init__(
        self,
        camera_ids: Sequence[str],
        labels: Sequence[str],
        *,
        canvas_size: Optional[Tuple[int, int]] = None,
        initial_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        point_radius: int = 18,
        fullscreen: bool = True,
    ) -> None:
        if not camera_ids:
            raise ValueError("camera_ids must contain at least one entry")
        if len(labels) != len(camera_ids):
            raise ValueError("labels length must match camera_ids length")

        if canvas_size is None and fullscreen:
            canvas_size = _get_screen_size()
        elif canvas_size is None:
            canvas_size = (1280, 720)

        self._width, self._height = canvas_size
        if self._width <= 0 or self._height <= 0:
            raise ValueError("canvas_size must contain positive values")

        self._point_radius = int(point_radius)
        self._markers: List[_Marker] = []
        self._window_name = "Camera Alignment"
        self._dragging: Optional[_Marker] = None
        self._last_drawn: Optional[np.ndarray] = None
        self._finish_requested = False
        self._dirty = True
        self._button_top_left = (self._width - 240, self._height - 70)
        self._button_bottom_right = (self._width - 40, self._height - 20)
        self._fullscreen_requested = fullscreen

        palette = [
            (0, 204, 255),
            (0, 255, 128),
            (255, 153, 51),
            (255, 102, 178),
            (128, 179, 255),
            (102, 255, 204),
            (255, 204, 102),
        ]

        defaults = initial_positions or {}
        generator = self._default_positions(len(camera_ids))
        for idx, cam_id in enumerate(camera_ids):
            label = labels[idx]
            if cam_id in defaults:
                norm_x, norm_y = defaults[cam_id]
            else:
                norm_x, norm_y = next(generator)
            color = palette[idx % len(palette)]
            self._markers.append(_Marker(cam_id, label, color, norm_x, norm_y))

        self._selected_index = 0
        self._markers[self._selected_index].is_selected = True

        window_flag = cv2.WINDOW_NORMAL
        cv2.namedWindow(self._window_name, window_flag)
        cv2.resizeWindow(self._window_name, self._width, self._height)
        if fullscreen:
            try:
                cv2.setWindowProperty(
                    self._window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
                # Force property to take effect immediately by showing a blank frame.
                blank = np.zeros((self._height, self._width, 3), dtype=np.uint8)
                cv2.imshow(self._window_name, blank)
                cv2.waitKey(1)
                cv2.moveWindow(self._window_name, 0, 0)
                # Query the effective window size after fullscreen adjustment
                try:
                    _, _, actual_w, actual_h = cv2.getWindowImageRect(self._window_name)
                    if actual_w > 0 and actual_h > 0:
                        self._width = actual_w
                        self._height = actual_h
                        self._button_top_left = (self._width - 240, self._height - 70)
                        self._button_bottom_right = (self._width - 40, self._height - 20)
                        self._background = self._build_background()
                        self._dirty = True
                except cv2.error:
                    pass
            except cv2.error:
                pass
        cv2.setMouseCallback(self._window_name, self._on_mouse)

        self._button_top_left = (self._width - 240, self._height - 70)
        self._button_bottom_right = (self._width - 40, self._height - 20)
        self._background = self._build_background()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _default_positions(self, count: int) -> Iterable[Tuple[float, float]]:
        if count == 1:
            yield 0.5, 0.1
            return
        spacing = 1.0 / (count + 1)
        for idx in range(count):
            norm_x = spacing * (idx + 1)
            norm_y = 0.15
            yield (norm_x, norm_y)

    def _marker_at(self, x: int, y: int) -> Optional[_Marker]:
        norm_x = x / self._width
        norm_y = y / self._height
        for marker in self._markers:
            px = marker.norm_x * self._width
            py = marker.norm_y * self._height
            if (px - x) ** 2 + (py - y) ** 2 <= (self._point_radius * 1.5) ** 2:
                return marker
        return None

    def _set_selected(self, marker: _Marker) -> None:
        for m in self._markers:
            m.is_selected = False
        marker.is_selected = True
        self._selected_index = self._markers.index(marker)
        self._dirty = True

    # ------------------------------------------------------------------
    # Drawing and interaction
    # ------------------------------------------------------------------

    def _build_background(self) -> np.ndarray:
        canvas = np.full((self._height, self._width, 3), 32, dtype=np.uint8)
        cv2.rectangle(canvas, (80, 80), (self._width - 80, self._height - 80), (55, 55, 55), 2)
        cv2.rectangle(canvas, (120, 120), (self._width - 120, self._height - 120), (75, 75, 75), 2)

        instructions = (
            "Drag markers with the mouse · Tab/Shift+Tab cycle selection · "
            "Arrow/hjkl ±0.01 · HJKL ±0.05 · S save snapshot · Q quit"
        )
        cv2.putText(
            canvas,
            instructions,
            (20, self._height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            canvas,
            self._button_top_left,
            self._button_bottom_right,
            (80, 80, 80),
            -1,
        )
        cv2.rectangle(
            canvas,
            self._button_top_left,
            self._button_bottom_right,
            (200, 200, 200),
            2,
        )
        button_label = "Continue"
        text_size = cv2.getTextSize(button_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = self._button_top_left[0] + (self._button_bottom_right[0] - self._button_top_left[0] - text_size[0]) // 2
        text_y = self._button_top_left[1] + (self._button_bottom_right[1] - self._button_top_left[1] + text_size[1]) // 2
        cv2.putText(
            canvas,
            button_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _draw(self) -> np.ndarray:
        canvas = self._background.copy()
        for marker in self._markers:
            px = int(round(marker.norm_x * self._width))
            py = int(round(marker.norm_y * self._height))
            cv2.line(canvas, (px, 0), (px, self._height - 1), marker.color, 1)
            cv2.line(canvas, (0, py), (self._width - 1, py), marker.color, 1)
            radius = self._point_radius + (4 if marker.is_selected else 0)
            cv2.circle(canvas, (px, py), radius, marker.color, -1)
            cv2.putText(
                canvas,
                marker.label,
                (px + radius + 8, py - radius - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"({marker.norm_x:.3f}, {marker.norm_y:.3f})",
                (px + radius + 8, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        self._last_drawn = canvas
        return canvas

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Optional[int]) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._button_top_left[0] <= x <= self._button_bottom_right[0] and self._button_top_left[1] <= y <= self._button_bottom_right[1]:
                self._finish_requested = True
                self._dirty = True
                return
            marker = self._marker_at(x, y)
            if marker is not None:
                self._set_selected(marker)
                self._dragging = marker
                marker.set_position(x / self._width, y / self._height)
                self._dirty = True
            else:
                self._dragging = None
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging is not None:
            self._dragging.set_position(x / self._width, y / self._height)
            self._dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging is not None:
                self._dragging.set_position(x / self._width, y / self._height)
                self._dragging = None
                self._select_relative(1)
                self._dirty = True
        # Some OpenCV builds do not provide EVENT_MOUSELEAVE; guard with getattr.
        elif event == getattr(cv2, "EVENT_MOUSELEAVE", -1):
            self._dragging = None

    # ------------------------------------------------------------------
    # Interaction loop
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        snapshot_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Tuple[float, float]]:
        while True:
            if self._fullscreen_requested:
                self._update_window_geometry()
            if self._dirty or self._last_drawn is None:
                frame = self._draw()
                cv2.imshow(self._window_name, frame)
                self._dirty = False
            key = cv2.waitKey(16) & 0xFFFF
            if self._finish_requested:
                break
            if key == 0xFFFF:
                continue
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                self._save_snapshot(snapshot_path)
                continue
            if key == ord("\t"):
                self._select_relative(1)
                continue
            if key == 353:  # Shift+Tab in some environments
                self._select_relative(-1)
                continue
            self._handle_key_move(key)

        positions = {marker.identifier: marker.position for marker in self._markers}

        if output_path is not None:
            payload = {
                "generated_at": time.time(),
                "canvas_size": [self._width, self._height],
                "markers": [
                    {
                        "id": marker.identifier,
                        "label": marker.label,
                        "norm_x": marker.norm_x,
                        "norm_y": marker.norm_y,
                    }
                    for marker in self._markers
                ],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        cv2.destroyWindow(self._window_name)
        cv2.waitKey(1)
        return positions

    def _save_snapshot(self, snapshot_path: Optional[Path]) -> None:
        if snapshot_path is None:
            print("Snapshot path not provided; skipping.")
            return
        if self._last_drawn is None:
            print("No frame rendered yet; skipping snapshot.")
            return
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot_path), self._last_drawn)
        print(f"Saved alignment snapshot to {snapshot_path}")

    def _select_relative(self, delta: int) -> None:
        self._markers[self._selected_index].is_selected = False
        self._selected_index = (self._selected_index + delta) % len(self._markers)
        self._markers[self._selected_index].is_selected = True
        self._dirty = True

    def _handle_key_move(self, key: int) -> None:
        marker = self._markers[self._selected_index]
        increment_small = 0.01
        increment_large = 0.05

        if key in (ord("h"), 81):  # left
            marker.set_position(marker.norm_x - increment_small, marker.norm_y)
            self._dirty = True
        elif key in (ord("l"), 83):  # right
            marker.set_position(marker.norm_x + increment_small, marker.norm_y)
            self._dirty = True
        elif key in (ord("k"), 82):  # up
            marker.set_position(marker.norm_x, marker.norm_y - increment_small)
            self._dirty = True
        elif key in (ord("j"), 84):  # down
            marker.set_position(marker.norm_x, marker.norm_y + increment_small)
            self._dirty = True
        elif key == ord("H"):
            marker.set_position(marker.norm_x - increment_large, marker.norm_y)
            self._dirty = True
        elif key == ord("L"):
            marker.set_position(marker.norm_x + increment_large, marker.norm_y)
            self._dirty = True
        elif key == ord("K"):
            marker.set_position(marker.norm_x, marker.norm_y - increment_large)
            self._dirty = True
        elif key == ord("J"):
            marker.set_position(marker.norm_x, marker.norm_y + increment_large)
            self._dirty = True

    def _update_window_geometry(self, *, force: bool = False) -> None:
        try:
            rect = cv2.getWindowImageRect(self._window_name)
        except cv2.error:
            return
        if not rect:
            return
        _, _, width, height = rect
        if width <= 0 or height <= 0:
            return
        if force or width != self._width or height != self._height:
            self._width = int(width)
            self._height = int(height)
            self._button_top_left = (self._width - 240, self._height - 70)
            self._button_bottom_right = (self._width - 40, self._height - 20)
            self._background = self._build_background()
            self._dirty = True


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def run_alignment_ui(
    camera_ids: Sequence[int | str],
    *,
    labels: Optional[Sequence[str]] = None,
    canvas_size: Optional[Tuple[int, int]] = None,
    initial_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    point_radius: int = 18,
    snapshot_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    fullscreen: bool = True,
) -> Dict[str, Tuple[float, float]]:
    ids = [str(identifier) for identifier in camera_ids]
    if labels is None:
        resolved_labels = [f"Camera {identifier}" for identifier in ids]
    else:
        if len(labels) != len(ids):
            raise ValueError("labels must match the number of camera identifiers")
        resolved_labels = list(labels)

    ui = _AlignmentCanvas(
        ids,
        resolved_labels,
        canvas_size=canvas_size,
        initial_positions=initial_positions,
        point_radius=point_radius,
        fullscreen=fullscreen,
    )
    return ui.run(snapshot_path=snapshot_path, output_path=output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive camera layout alignment tool.")
    parser.add_argument("--cameras", nargs="*", help="Identifiers (indices or names) for each camera.")
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional human-readable labels matching the cameras list.",
    )
    parser.add_argument("--canvas-width", type=int, help="Alignment canvas width in pixels.")
    parser.add_argument("--canvas-height", type=int, help="Alignment canvas height in pixels.")
    parser.add_argument("--point-radius", type=int, default=18, help="Marker radius in pixels (default: 18).")
    parser.add_argument("--snapshot", type=Path, help="Optional path for saving a snapshot of the layout.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to write the final positions.")
    parser.add_argument(
        "--initial",
        type=Path,
        help="Optional JSON file containing previous alignment results to preload positions.",
    )
    parser.add_argument(
        "--no-fullscreen",
        action="store_true",
        help="Disable fullscreen mode for the alignment window.",
    )
    args = parser.parse_args(argv)

    if not args.cameras:
        parser.error("No cameras specified. Provide identifiers after --cameras.")

    if args.labels and len(args.labels) != len(args.cameras):
        parser.error("--labels must provide the same number of entries as --cameras.")

    initial_positions: Optional[Dict[str, Tuple[float, float]]] = None
    if args.initial:
        initial_path = args.initial.expanduser().resolve()
        try:
            payload = json.loads(initial_path.read_text(encoding="utf-8"))
            markers = payload.get("markers", [])
            initial_positions = {
                str(entry["id"]): (float(entry["norm_x"]), float(entry["norm_y"]))
                for entry in markers
                if "id" in entry
            }
        except FileNotFoundError:
            print(f"Warning: initial positions file not found: {initial_path}", file=sys.stderr)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"Warning: failed to load initial positions ({exc}); starting fresh.", file=sys.stderr)

    custom_canvas: Optional[Tuple[int, int]] = None
    if args.canvas_width or args.canvas_height:
        if args.canvas_width is None or args.canvas_height is None:
            parser.error("--canvas-width and --canvas-height must be provided together.")
        custom_canvas = (int(args.canvas_width), int(args.canvas_height))

    positions = run_alignment_ui(
        args.cameras,
        labels=args.labels,
        canvas_size=custom_canvas,
        initial_positions=initial_positions,
        point_radius=args.point_radius,
        snapshot_path=args.snapshot,
        output_path=args.output,
        fullscreen=not args.no_fullscreen,
    )

    for cam_id, (norm_x, norm_y) in positions.items():
        print(f"{cam_id}: norm=({norm_x:.4f}, {norm_y:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
