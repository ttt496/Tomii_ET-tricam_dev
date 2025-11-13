from abc import ABC, abstractmethod
from typing import Optional, List, Sequence, Tuple
from dataclasses import dataclass

from core.dtypes.media import ImageArray, FrameID, Timestamp
from core.dtypes.person import PersonID
from core.integrators.face_detector.dtypes.bounding import FaceBoundingBox, EyeBoundingBox

@dataclass
class BlinkDetectorInput:
    image: ImageArray
    faces: Sequence[FaceBoundingBox]
    person_ids: Optional[List[PersonID]] = None
    frame_id: Optional[FrameID] = None
    timestamp: Optional[Timestamp] = None


@dataclass
class EARBlinkDetails:
    person_id: PersonID
    left_eye_bb: Optional[EyeBoundingBox] = None
    right_eye_bb: Optional[EyeBoundingBox] = None
    left_eye_image: Optional[ImageArray] = None
    right_eye_image: Optional[ImageArray] = None
    left_eye_center_frame: Optional[Tuple[float, float]] = None
    right_eye_center_frame: Optional[Tuple[float, float]] = None


@dataclass 
class BlinkResult:
    person_id: Optional[PersonID] = None
    blink_probability: Optional[float] = None
    is_blink: Optional[bool] = None
    ear_result: Optional[EARBlinkDetails] = None


@dataclass
class BlinkDetectorResult:
    results: Sequence[BlinkResult]
    processing_time: float = 0.0
    

class BlinkDetector(ABC):
    @abstractmethod
    def __call__(self, input: BlinkDetectorInput) -> BlinkDetectorResult:
        pass
    
    @abstractmethod
    def __enter__(self):
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
