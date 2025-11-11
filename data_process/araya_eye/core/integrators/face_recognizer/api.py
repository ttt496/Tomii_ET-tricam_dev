from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

from core.dtypes.media import ImageArray
from core.dtypes.person import PersonID
from ..face_detector.dtypes.bounding import FaceBoundingBox


@dataclass
class FaceRecognizerInput:
    image: ImageArray
    faces: List[FaceBoundingBox]


@dataclass
class FaceResult:
    person_id: PersonID
    confidence: float


@dataclass
class FaceRecognizerResult:
    results: List[FaceResult]
    processing_time: float


class FaceRecognizer(ABC):
    @abstractmethod
    def __call__(self, input: FaceRecognizerInput) -> FaceRecognizerResult:
        pass
    
    @abstractmethod
    def __enter__(self):
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass