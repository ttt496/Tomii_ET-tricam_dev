import os
import sys
import warnings
import contextlib
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, List
from .config import MpFacemeshConfig

"""MediaPipe使用時の警告を完全に抑制"""
# TensorFlow/TensorFlow Lite警告抑制
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# abseil-cpp / glog警告抑制（MediaPipeのC++層）
os.environ.setdefault('ABSL_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('GLOG_v', '0')
os.environ.setdefault('GLOG_logtostderr', '0')
os.environ.setdefault('GLOG_alsologtostderr', '0')

# MediaPipe固有警告のPython warnings抑制
warnings.filterwarnings('ignore', message='.*Feedback manager requires.*')
warnings.filterwarnings('ignore', message='.*single signature inference.*')
warnings.filterwarnings('ignore', message='.*inference_feedback_manager.*')

# stderr抑制してMediaPipeインポート
with contextlib.redirect_stderr(open(os.devnull, 'w')):
    import mediapipe as mp

@dataclass
class Landmarks:
    @dataclass
    class Position:
        x: int
        y: int 
    landmark: List[Position]

class MpFacemeshModel:
    """MediaPipe FaceMesh model"""
    
    def __init__(self, config: MpFacemeshConfig):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore
        self.face_mesh: Optional[Any] = None
    
    def __enter__(self):
        """Initialize resources"""
        try:
            # MediaPipe初期化時も警告を抑制
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=self.config.max_num_faces,
                    refine_landmarks=self.config.refine_landmarks,
                    min_detection_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence
                )
            return self
        except Exception as e:
            raise RuntimeError(f"FaceMesh model initialization failed: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
    
    def __call__(self, rgb_face_image: np.ndarray) -> Optional[Landmarks]:
        """
        FaceMesh landmark detection
        
        Args:
            rgb_face_image: RGB format face image
            
        Returns:
            MediaPipe landmark object or None
        """
        if not self.face_mesh:
            raise RuntimeError("Model not initialized. Use with context manager.")
        
        try:
            # MediaPipe処理時も警告を抑制
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                results = self.face_mesh.process(rgb_face_image)
            
            if not results.multi_face_landmarks:
                return None
            
            # Return first face landmarks
            return results.multi_face_landmarks[0]
            
        except Exception as e:
            raise RuntimeError(f"FaceMesh processing failed: {e}")