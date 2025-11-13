from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

from core.models.yolov9_wb25.config import YoloV9Wb25Config

@pydantic_dataclass
@dataclass
class YoloV9Wb25FaceDetectorConfig(YoloV9Wb25Config): pass