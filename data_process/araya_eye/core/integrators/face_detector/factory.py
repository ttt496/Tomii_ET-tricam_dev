from .api import FaceDetector
from .config import FaceDetectorConfig

def create_face_detector(config: FaceDetectorConfig)->FaceDetector:
    
    if config.type == "yolov9_wb25":
        from .wrappers.yolov9_wb25.wrapper import YoloV9Wb25FaceDetector
        assert config.yolov9_wb25
        return YoloV9Wb25FaceDetector(config.yolov9_wb25)
    
    raise ValueError(f"Unknown face detector type: {config.type}")