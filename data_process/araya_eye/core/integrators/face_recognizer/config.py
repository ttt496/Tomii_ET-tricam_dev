from typing import Literal, Optional, Any
from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .wrappers.insight_face.config import InsightFaceRecognizerConfig
from .wrappers.iou_tracker.config import IouTrackerConfig


@pydantic_dataclass
@dataclass
class FaceRecognizerConfig:
    type: Literal["insight_face", "iou_tracker"]
    insight_face: Optional[InsightFaceRecognizerConfig] = None
    iou_tracker: Optional[IouTrackerConfig] = None