from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

from core.models.lstm_base_01 import LstmBase01Config
from core.models.mp_facemesh import MpFacemeshConfig

@pydantic_dataclass
@dataclass
class LstmBase01BlinkDetectorConfig:
    mp_facemesh: MpFacemeshConfig
    lstm_base_01: LstmBase01Config