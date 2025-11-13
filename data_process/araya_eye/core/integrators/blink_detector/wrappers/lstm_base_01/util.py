from typing import Literal, Optional, Tuple, List
import numpy as np
from core.models.mp_facemesh.model import Landmarks
from core.integrators.face_detector.dtypes.bounding import BoundingBox, EyeBoundingBox

def extract_eye_bb(landmarks:Landmarks, eye_side:Literal["left","right"])->EyeBoundingBox:
    """統合されたEAR計算とランドマーク抽出"""
    # MediaPipe FaceMesh正規ランドマークインデックス（参考実装準拠）
    if eye_side == "left": eye_indices = [263, 387, 385, 362, 380, 373]
    if eye_side == "right": eye_indices = [33, 160, 158, 133, 153, 144]
    
    # Extract eye landmarks for both EAR calculation and visualization
    eye_points:List[Tuple[float,float]] = []
    
    min_x, max_x, min_y, max_y = None, None, None, None
    for idx in eye_indices:
        landmark = landmarks.landmark[idx]
        x, y = float(landmark.x), float(landmark.y)
        if min_x == None or min_x > x: min_x = x
        if max_x == None or max_x < x: max_x = x
        if min_y == None or min_y > y: min_y = y
        if max_y == None or max_y < y: max_y = y
        eye_points.append((x, y))
    
    if len(eye_points) != 6:
        raise Exception(f"WARNING: Invalid eye landmarks count: {len(eye_points)} (expected 6)")
    assert min_x != None and max_x != None and min_y != None and max_y != None
    
    return EyeBoundingBox(min_x, min_y, max_x-min_x, max_y-min_y, landmarks=eye_points)

    
# def validate_eye_position(mp_eye_bb: EyeBoundingBox, face_eye_bb: BoundingBox) -> bool:
#     # mp_eye_bb: mediapipe_facemeshから計算される eye_bounding_box
#     # face_eye_bb: face_detectorの bounding_box
    
#     # MediaPipe eye bounding boxの中心点
#     mp_center_x = mp_eye_bb.x + mp_eye_bb.width / 2
#     mp_center_y = mp_eye_bb.y + mp_eye_bb.height / 2
    
#     # Face detector eye bounding boxの中心点  
#     face_center_x = face_eye_bb.x + face_eye_bb.width / 2
#     face_center_y = face_eye_bb.y + face_eye_bb.height / 2
    
#     # 距離計算
#     distance_x = abs(mp_center_x - face_center_x)
#     distance_y = abs(mp_center_y - face_center_y)
    
#     # 許容範囲（face_eye_bbを基準に1.5倍）
#     allowed_x = face_eye_bb.width * 1.5 / 2
#     allowed_y = face_eye_bb.height * 1.5 / 2
    
#     within_x = distance_x <= allowed_x
#     within_y = distance_y <= allowed_y
    
#     return within_x and within_y