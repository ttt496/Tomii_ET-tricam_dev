"""Eye tracking utilities for extracting eye regions from webcam frames."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class EyeRegion:
    """Container describing an eye crop extracted from a frame."""

    bbox: Tuple[int, int, int, int]
    image: np.ndarray


@dataclass
class FaceEyes:
    """Detected face bounding box with associated eye crops."""

    face_bbox: Tuple[int, int, int, int]
    eyes: List[EyeRegion]


class EyeTracker:
    """Detect faces and extract eye regions using Haar cascade classifiers."""

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_eye_neighbors: int = 4,
        min_face_size: Tuple[int, int] = (80, 80),
        min_eye_size: Tuple[int, int] = (24, 16),
    ) -> None:
        face_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        eye_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"

        self._face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
        self._eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))
        if self._face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade: {face_cascade_path}")
        if self._eye_cascade.empty():
            raise RuntimeError(f"Failed to load eye cascade: {eye_cascade_path}")

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_eye_neighbors = min_eye_neighbors
        self._min_face_size = min_face_size
        self._min_eye_size = min_eye_size

    def detect(self, frame: np.ndarray) -> List[FaceEyes]:
        """Detect faces and eyes in the provided frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_face_size,
        )

        detections: List[FaceEyes] = []
        for (x, y, w, h) in faces:
            face_roi_gray = gray[y : y + h, x : x + w]
            face_roi_color = frame[y : y + h, x : x + w]

            eye_candidates = self._eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.05,
                minNeighbors=self._min_eye_neighbors,
                minSize=self._min_eye_size,
            )

            eyes: List[EyeRegion] = []
            for (ex, ey, ew, eh) in eye_candidates:
                eye_img = face_roi_color[ey : ey + eh, ex : ex + ew]
                eyes.append(
                    EyeRegion(
                        bbox=(x + ex, y + ey, ew, eh),
                        image=eye_img,
                    )
                )

            detections.append(FaceEyes(face_bbox=(x, y, w, h), eyes=eyes))
        return detections

    def annotate(self, frame: np.ndarray, detections: Iterable[FaceEyes]) -> np.ndarray:
        """Draw face and eye bounding boxes on a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det.face_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for eye in det.eyes:
                ex, ey, ew, eh = eye.bbox
                cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (0, 128, 255), 2)
        return annotated


def _stack_eye_views(eyes: List[EyeRegion], *, max_cols: int = 4) -> Optional[np.ndarray]:
    if not eyes:
        return None

    resized = [cv2.resize(eye.image, (120, 80)) for eye in eyes]
    rows = []
    for idx in range(0, len(resized), max_cols):
        row_imgs = resized[idx : idx + max_cols]
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)


def run_eye_tracking(
    camera_index: int = 0,
    *,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_eye_neighbors: int = 4,
    min_face_size: Tuple[int, int] = (80, 80),
    min_eye_size: Tuple[int, int] = (24, 16),
    preview_scale: float = 1.0,
    max_seconds: Optional[float] = None,
) -> None:
    tracker = EyeTracker(
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_eye_neighbors=min_eye_neighbors,
        min_face_size=min_face_size,
        min_eye_size=min_eye_size,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}")

    window_name = "Eye Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if preview_scale != 1.0:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * preview_scale)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * preview_scale)
        if frame_width > 0 and frame_height > 0:
            cv2.resizeWindow(window_name, frame_width, frame_height)

    eye_window = "Eye Crops"
    cv2.namedWindow(eye_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window, 480, 240)

    start = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera stopped delivering frames")

            detections = tracker.detect(frame)
            annotated = tracker.annotate(frame, detections)

            cv2.imshow(window_name, annotated)

            eyes: List[EyeRegion] = []
            for det in detections:
                eyes.extend(det.eyes)
            eye_stack = _stack_eye_views(eyes)
            if eye_stack is not None:
                cv2.imshow(eye_window, eye_stack)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if max_seconds is not None:
                elapsed_seconds = (cv2.getTickCount() - start) / freq
                if elapsed_seconds >= max_seconds:
                    break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        cv2.destroyWindow(eye_window)
        cv2.waitKey(1)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye tracking preview using Haar cascades.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open (default: 0)")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Face detection scale factor")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Face detection minNeighbors")
    parser.add_argument("--min-eye-neighbors", type=int, default=4, help="Eye detection minNeighbors")
    parser.add_argument("--min-face-size", nargs=2, type=int, metavar=("W", "H"), help="Minimum face size")
    parser.add_argument("--min-eye-size", nargs=2, type=int, metavar=("W", "H"), help="Minimum eye size")
    parser.add_argument("--preview-scale", type=float, default=1.0, help="Scale preview window")
    parser.add_argument("--max-seconds", type=float, help="Limit preview duration")
    return parser.parse_args(argv)


def _tuple_arg(values: Optional[Sequence[int]], default: Tuple[int, int]) -> Tuple[int, int]:
    if values is None:
        return default
    if len(values) != 2:
        raise ValueError("Expected two integers for size arguments")
    return int(values[0]), int(values[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        run_eye_tracking(
            camera_index=args.camera,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_eye_neighbors=args.min_eye_neighbors,
            min_face_size=_tuple_arg(args.min_face_size, (80, 80)),
            min_eye_size=_tuple_arg(args.min_eye_size, (24, 16)),
            preview_scale=args.preview_scale,
            max_seconds=args.max_seconds,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
