from typing import Literal, Optional
from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .wrappers.lstm_base_01.config import LstmBase01BlinkDetectorConfig


@pydantic_dataclass
@dataclass
class BlinkDetectorConfig:
    type: Literal["lstm_base_01"]
    lstm_base_01_blink_detector: Optional[LstmBase01BlinkDetectorConfig]