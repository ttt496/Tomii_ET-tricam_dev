from .api import FaceRecognizer
from .config import FaceRecognizerConfig

def create_face_recognizer(config: FaceRecognizerConfig) -> FaceRecognizer:
    
    if config.type == "insight_face":
        from .wrappers.insight_face.wrapper import InsightFaceRecognizer
        assert config.insight_face
        return InsightFaceRecognizer(config.insight_face)
    
    elif config.type == "iou_tracker":
        from .wrappers.iou_tracker.wrapper import IouTracker
        assert config.iou_tracker
        return IouTracker(config.iou_tracker)
    
    raise ValueError(f"Unknown face recognizer type: {config.type}")