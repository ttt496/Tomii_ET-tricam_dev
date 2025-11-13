"""
YOLO Face Detector Wrapper Implementation
YOLOモデルを使用した顔検出器の実装
"""
import time
from typing import List
from pathlib import Path
from core.models.yolov9_wb25 import YoloV9Wb25Model, YoloV9Wb25Config, YoloV9Wb25Box
from ...api import FaceDetector, FaceDetectorInput, FaceDetectorResult
from ...dtypes.bounding import FaceBoundingBox, FaceEyeBoundingBox, EyeBoundingBox
from .config import YoloV9Wb25FaceDetectorConfig

class YoloV9Wb25FaceDetector(FaceDetector):
    def __init__(self, config: YoloV9Wb25FaceDetectorConfig):
        
        self.config = config
        self.yolo_model = YoloV9Wb25Model(YoloV9Wb25Config(
            model_path=config.model_path,
            object_score_threshold=config.object_score_threshold,
            attribute_score_threshold=config.attribute_score_threshold,
            execution_provider=config.execution_provider,
            input_size=config.input_size,
            nms_threshold=config.nms_threshold
        ))
        
        print(f"INFO: YOLO face detector ready: {Path(self.config.model_path).name} @ {self.config.input_size}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def __call__(self, input: FaceDetectorInput) -> FaceDetectorResult:
        start_time = time.time()
        image = input.image
        boxes = self.yolo_model(image)
        boxes = list(filter(lambda box:box.confidence >= self.config.object_score_threshold, boxes))
        head_boxes = [box for box in boxes if box.class_name == "head"]
        eye_boxes = [box for box in boxes if box.class_name == "eye"]
        
        faces:List[FaceBoundingBox] = []        
        for head_box in head_boxes:
            x = max(0, head_box.x)
            y = max(0, head_box.y) 
            width = max(0, head_box.width)
            height = max(0, head_box.height)
            
            def is_eye_in_head(eye_box, head_box) -> bool:
                """眼がhead内に含まれているかチェック"""
                eye_center_x = eye_box.x + eye_box.width / 2
                eye_center_y = eye_box.y + eye_box.height / 2
                return (head_box.x <= eye_center_x <= head_box.x + head_box.width and
                        head_box.y <= eye_center_y <= head_box.y + head_box.height)
            # このheadに含まれるeyeを探す
            contained_eyes = [eye for eye in eye_boxes if is_eye_in_head(eye, head_box)]
            
            if contained_eyes:
                def to_eye_box(yolo_box: YoloV9Wb25Box) -> EyeBoundingBox:
                    """YOLOボックスを目ボックスに変換"""
                    return EyeBoundingBox(
                        x=yolo_box.x, y=yolo_box.y, width=yolo_box.width, height=yolo_box.height,
                        confidence=yolo_box.confidence
                    )
                # 左右の目を位置で判定
                if len(contained_eyes) == 1:
                    # 1個の場合は中央を基準に判定
                    eye = contained_eyes[0]
                    eye_center_x = eye.x + eye.width / 2
                    head_center_x = head_box.x + head_box.width / 2
                    if eye_center_x < head_center_x:
                        left_eye = to_eye_box(eye)
                        right_eye = None
                    else:
                        left_eye = None
                        right_eye = to_eye_box(eye)
                elif len(contained_eyes) >= 2:
                    # 2個以上は左右で分ける
                    eyes_sorted = sorted(contained_eyes, key=lambda e: e.x + e.width / 2)
                    left_eye = to_eye_box(eyes_sorted[0])  # 左側
                    right_eye = to_eye_box(eyes_sorted[1])  # 右側
                else:
                    left_eye = right_eye = None
                face_box = FaceEyeBoundingBox(
                    x=x, y=y, width=width, height=height,
                    confidence=head_box.confidence,
                    left_eye=left_eye,
                    right_eye=right_eye
                )
            else:
                # 目がない場合はFaceBoundingBox作成
                face_box = FaceBoundingBox(
                    x=x, y=y, width=width, height=height,
                    confidence=head_box.confidence
                )
            faces.append(face_box)

        return FaceDetectorResult(faces=faces, processing_time=time.time() - start_time)