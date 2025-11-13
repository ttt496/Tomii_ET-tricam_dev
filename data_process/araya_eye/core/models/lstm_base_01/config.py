from typing import Literal
from dataclasses import dataclass

@dataclass
class LstmBase01Config:
    model_path: str
    config_path: str
    device: Literal["cpu", "cuda"]
    dropout: float
    blink_threshold: float
    person_retention_frames: int