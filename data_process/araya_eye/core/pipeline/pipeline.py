from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from core.dtypes.media import FrameID, ImageArray, Timestamp
from core.dtypes.person import PersonID
from core.integrators.face_detector import (
    FaceDetectorInput,
    FaceDetectorResult,
)
from core.integrators.face_detector.dtypes.bounding import FaceBoundingBox
from core.integrators.face_recognizer import (
    FaceRecognizerInput,
    FaceRecognizerResult,
    FaceResult,
)
from core.integrators.face_detector.factory import create_face_detector
from core.integrators.face_recognizer.factory import create_face_recognizer
from core.integrators.blink_detector import EARBlinkDetails
from core.integrators.blink_detector.wrappers.lstm_base_01.mediapipe_processor import (
    process_mediapipe,
)
from core.models.mp_facemesh import MpFacemeshModel
from .config import PipelineConfig


@dataclass
class MediapipeStageResult:
    """MediaPipe eye extraction output for a single frame."""

    ear_details: Sequence[EARBlinkDetails]
    processing_time: float


@dataclass
class PipelineFaceState:
    """Aggregated view for a single detected face within a frame."""

    bbox: FaceBoundingBox
    recognition: Optional[FaceResult] = None
    eye_details: Optional[EARBlinkDetails] = None


@dataclass
class PipelineFrameResult:
    """Container that exposes both per-stage outputs and merged per-face states."""

    faces: Sequence[PipelineFaceState]
    face_detector: FaceDetectorResult
    face_recognizer: FaceRecognizerResult
    mediapipe: MediapipeStageResult

    @property
    def total_processing_time(self) -> float:
        """Best-effort aggregation of each stage's processing time."""
        total = 0.0
        if self.face_detector.processing_time:
            total += self.face_detector.processing_time
        total += self.face_recognizer.processing_time
        total += self.mediapipe.processing_time
        return total


class EyeCropPipeline:
    """Runs face detection → recognition → MediaPipe eye extraction per frame."""

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._face_detector = create_face_detector(config.face_detector)
        self._face_recognizer = create_face_recognizer(config.face_recognizer)
        self._facemesh_model = MpFacemeshModel(config.mp_facemesh)

    def __enter__(self) -> "EyeCropPipeline":
        self._face_detector.__enter__()
        self._face_recognizer.__enter__()
        self._facemesh_model.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._facemesh_model.__exit__(exc_type, exc_val, exc_tb)
        self._face_recognizer.__exit__(exc_type, exc_val, exc_tb)
        self._face_detector.__exit__(exc_type, exc_val, exc_tb)

    def __call__(
        self,
        image: ImageArray,
        *,
        frame_id: Optional[FrameID] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> PipelineFrameResult:
        return self.process(image, frame_id=frame_id, timestamp=timestamp)

    def process(
        self,
        image: ImageArray,
        *,
        frame_id: Optional[FrameID] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> PipelineFrameResult:
        face_result = self._face_detector(FaceDetectorInput(image=image))
        faces = list(face_result.faces)

        recognition_result = self._run_face_recognizer(image, faces)
        person_ids = self._extract_person_ids(len(faces), recognition_result)
        mediapipe_result = self._run_mediapipe(
            image=image,
            faces=faces,
            person_ids=person_ids,
        )

        merged_faces = self._merge_face_states(faces, recognition_result, mediapipe_result)

        return PipelineFrameResult(
            faces=merged_faces,
            face_detector=face_result,
            face_recognizer=recognition_result,
            mediapipe=mediapipe_result,
        )

    def _run_face_recognizer(
        self,
        image: ImageArray,
        faces: List[FaceBoundingBox],
    ) -> FaceRecognizerResult:
        if not faces:
            return FaceRecognizerResult(results=[], processing_time=0.0)

        return self._face_recognizer(
            FaceRecognizerInput(image=image, faces=faces)
        )

    def _run_mediapipe(
        self,
        *,
        image: ImageArray,
        faces: List[FaceBoundingBox],
        person_ids: List[PersonID],
    ) -> MediapipeStageResult:
        if not faces:
            return MediapipeStageResult(ear_details=[], processing_time=0.0)

        start = time.time()
        ear_details = process_mediapipe(
            image=image,
            faces=faces,
            facemesh_model=self._facemesh_model,
            person_ids=person_ids,
        )
        processing_time = time.time() - start

        return MediapipeStageResult(
            ear_details=ear_details,
            processing_time=processing_time,
        )

    @staticmethod
    def _extract_person_ids(
        expected_count: int, recognition_result: FaceRecognizerResult
    ) -> List[PersonID]:
        ids = [res.person_id for res in recognition_result.results]
        if len(ids) < expected_count:
            ids.extend([PersonID(-1)] * (expected_count - len(ids)))
        return ids[:expected_count]

    @staticmethod
    def _merge_face_states(
        faces: Sequence[FaceBoundingBox],
        recognition_result: FaceRecognizerResult,
        mediapipe_result: MediapipeStageResult,
    ) -> List[PipelineFaceState]:
        face_states: List[PipelineFaceState] = []
        for idx, bbox in enumerate(faces):
            recognition = (
                recognition_result.results[idx]
                if idx < len(recognition_result.results)
                else None
            )
            eye_details = (
                mediapipe_result.ear_details[idx]
                if idx < len(mediapipe_result.ear_details)
                else None
            )
            face_states.append(
                PipelineFaceState(
                    bbox=bbox,
                    recognition=recognition,
                    eye_details=eye_details,
                )
            )
        return face_states
