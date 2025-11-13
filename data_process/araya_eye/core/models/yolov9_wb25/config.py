from typing import Literal, Tuple
from dataclasses import dataclass

@dataclass
class YoloV9Wb25Config:
    model_path: str
    object_score_threshold: float
    attribute_score_threshold: float
    execution_provider: Literal["cpu", "cuda"]
    input_size: Tuple[int, int]
    nms_threshold: float