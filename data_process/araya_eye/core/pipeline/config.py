from __future__ import annotations

from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

from core.integrators.face_detector import FaceDetectorConfig
from core.integrators.face_recognizer import FaceRecognizerConfig
from core.models.mp_facemesh import MpFacemeshConfig


@pydantic_dataclass
@dataclass
class PipelineConfig:
    """Top-level configuration that wires together the perception stages."""

    face_detector: FaceDetectorConfig
    face_recognizer: FaceRecognizerConfig
    mp_facemesh: MpFacemeshConfig
