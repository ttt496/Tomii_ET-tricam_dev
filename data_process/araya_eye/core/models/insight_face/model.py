import numpy as np
from typing import List, Optional
import insightface

from core.dtypes.media import ImageArray
from core.integrators.face_detector.dtypes.bounding import FaceBoundingBox
from .config import InsightFaceConfig

class InsightFaceModel:
    def __init__(self, config: InsightFaceConfig):
        self.config = config
        self.app = None
        self._initialized = False
    
    def initialize(self) -> bool:
        if self._initialized: return True
            
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.execution_provider == 'cuda' else ['CPUExecutionProvider']
            
            self.app = insightface.app.FaceAnalysis(name=self.config.model_name, providers=providers)
            ctx_id = 0 if self.config.execution_provider == 'cuda' else -1
            self.app.prepare(ctx_id=ctx_id, det_size=self.config.det_size)
            
            self._initialized = True
            return True
            
        except ImportError:
            print("WARNING: InsightFace not available")
            return False
        except Exception as e:
            print(f"ERROR: InsightFace initialization failed: {e}")
            return False
    
    def extract_embeddings(self, image: ImageArray, faces: List[FaceBoundingBox]) -> List[Optional[np.ndarray]]:
        if not self.initialize(): return [None] * len(faces)
        
        embeddings = []
        for face_bbox in faces:
            try:
                x = int(float(face_bbox.x))
                y = int(float(face_bbox.y))
                w = int(float(face_bbox.width))
                h = int(float(face_bbox.height))
                
                # 境界チェック
                img_h, img_w = image.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = max(1, min(w, img_w - x))
                h = max(1, min(h, img_h - y))
                
                face_region = image[y:y+h, x:x+w]
                
                # InsightFace処理
                assert self.app
                detected_faces = self.app.get(face_region)
                if detected_faces:
                    # 最も信頼度の高い顔を使用
                    best_face = max(detected_faces, key=lambda f: f.det_score)
                    embeddings.append(best_face.embedding)
                else:
                    embeddings.append(None)
                    
            except Exception as e:
                print(f"WARNING: Embedding extraction failed: {e}")
                embeddings.append(None)
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """コサイン類似度計算"""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0: return 0.0
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return max(0.0, similarity)
            
        except Exception:
            return 0.0