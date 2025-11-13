from typing import Optional, Literal
from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .wrappers.yolov9_wb25.config import YoloV9Wb25FaceDetectorConfig


@pydantic_dataclass
@dataclass
class FaceDetectorConfig:
    type: Literal["yolov9_wb25"]
    yolov9_wb25: Optional[YoloV9Wb25FaceDetectorConfig]