from typing import Literal, Tuple
from dataclasses import dataclass


@dataclass
class InsightFaceConfig:
    execution_provider: Literal["cpu", "cuda"]
    model_name: Literal["buffalo_l", "buffalo_m", "buffalo_s"]
    det_size: Tuple[int, int]
    recognition_threshold: float