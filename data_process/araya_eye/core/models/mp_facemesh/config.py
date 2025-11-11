from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

@pydantic_dataclass
@dataclass
class MpFacemeshConfig:
    max_num_faces: int
    refine_landmarks: bool
    min_detection_confidence: float
    min_tracking_confidence: float