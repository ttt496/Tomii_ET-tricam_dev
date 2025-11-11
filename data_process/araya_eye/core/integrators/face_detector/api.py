from abc import ABC, abstractmethod
from typing import Sequence, Optional
from dataclasses import dataclass

from core.dtypes.media import ImageArray
from .dtypes.bounding import FaceBoundingBox

@dataclass
class FaceDetectorInput:
    image: ImageArray


@dataclass
class FaceDetectorResult:
    faces: Sequence[FaceBoundingBox]
    processing_time: Optional[float] = None


class FaceDetector(ABC):
    @abstractmethod
    def __call__(self, input: FaceDetectorInput) -> FaceDetectorResult:
        pass
    
    @abstractmethod
    def __enter__(self):
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass