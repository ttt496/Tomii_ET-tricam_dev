from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.dtypes.media import ImageArray
from core.dtypes.person import PersonID
from core.integrators.face_detector.dtypes.bounding import FaceBoundingBox, EyeBoundingBox
from core.models.mp_facemesh.model import MpFacemeshModel

from ...api import EARBlinkDetails
from .util import extract_eye_bb

EYE_CROP_WIDTH = 140
EYE_CROP_HEIGHT = 40


def process_mediapipe(
    image: ImageArray,
    faces: Sequence[FaceBoundingBox],
    facemesh_model: MpFacemeshModel,
    person_ids: Optional[List[PersonID]],
    padding_ratio: float = 0.1,
) -> List[EARBlinkDetails]:
    """
    Runs MediaPipe FaceMesh on each face, extracts eye bounding boxes,
    and returns EAR metadata enriched with cropped left/right eye images.
    """
    ear_results: List[EARBlinkDetails] = []
    if not faces:
        return ear_results

    person_ids = person_ids or list(range(len(faces)))
    for person_id, face_bbox in zip(person_ids, faces):
        x1 = max(0, int(face_bbox.x))
        y1 = max(0, int(face_bbox.y))
        x2 = max(x1, min(image.shape[1], int(face_bbox.x + face_bbox.width)))
        y2 = max(y1, min(image.shape[0], int(face_bbox.y + face_bbox.height)))
        face_image = image[y1:y2, x1:x2]

        if face_image.size == 0:
            ear_results.append(EARBlinkDetails(person_id=person_id))
            continue

        face_rgb = (
            face_image[:, :, ::-1]
            if len(face_image.shape) == 3 and face_image.shape[2] == 3
            else face_image
        )

        landmarks = facemesh_model(face_rgb)
        if not landmarks:
            ear_results.append(EARBlinkDetails(person_id=person_id))
            continue

        left_eye_bb = extract_eye_bb(landmarks, "left")
        right_eye_bb = extract_eye_bb(landmarks, "right")

        left_eye_image = _crop_eye_image(face_image, left_eye_bb, padding_ratio)
        right_eye_image = _crop_eye_image(face_image, right_eye_bb, padding_ratio)
        left_center = _eye_center_in_frame(left_eye_bb, (x1, y1), face_image.shape[1], face_image.shape[0])
        right_center = _eye_center_in_frame(right_eye_bb, (x1, y1), face_image.shape[1], face_image.shape[0])

        ear_results.append(
            EARBlinkDetails(
                person_id=person_id,
                left_eye_bb=left_eye_bb,
                right_eye_bb=right_eye_bb,
                left_eye_image=left_eye_image,
                right_eye_image=right_eye_image,
                left_eye_center_frame=left_center,
                right_eye_center_frame=right_center,
            )
        )

    return ear_results


def _crop_eye_image(
    face_image: ImageArray,
    eye_bb: Optional[EyeBoundingBox],
    padding_ratio: float,
) -> Optional[ImageArray]:
    if eye_bb is None or face_image.size == 0:
        return None

    height, width = face_image.shape[:2]

    def _scale(value: float, size: int) -> float:
        """MediaPipe landmarks are normalized [0, 1], but fall back to pixels if >1."""
        if value <= 1.0:
            return value * size
        return value

    x1 = _scale(eye_bb.x, width)
    y1 = _scale(eye_bb.y, height)
    x2 = x1 + _scale(eye_bb.width, width)
    y2 = y1 + _scale(eye_bb.height, height)

    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio

    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(width, int(np.ceil(x2 + pad_x)))
    y2 = min(height, int(np.ceil(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None

    eye_crop = face_image[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None
    resized = cv2.resize(
        eye_crop,
        (EYE_CROP_WIDTH, EYE_CROP_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _eye_center_in_frame(
    eye_bb: Optional[EyeBoundingBox],
    face_origin: Tuple[int, int],
    face_width: int,
    face_height: int,
) -> Optional[Tuple[float, float]]:
    if eye_bb is None:
        return None

    def _scale(value: float, size: int) -> float:
        if value <= 1.0:
            return value * size
        return value

    center_x = eye_bb.x + eye_bb.width / 2.0
    center_y = eye_bb.y + eye_bb.height / 2.0
    center_px = _scale(center_x, face_width)
    center_py = _scale(center_y, face_height)
    frame_x = face_origin[0] + center_px
    frame_y = face_origin[1] + center_py
    return (float(frame_x), float(frame_y))
