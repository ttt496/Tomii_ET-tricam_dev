from typing import Optional, cast, Tuple, List, Union
from dataclasses import dataclass
import numpy as np

from core.dtypes.bounding import BoundingBox, FloatArray

@dataclass
class FaceBoundingBox(BoundingBox): pass

EARValue = float

@dataclass
class EyeBoundingBox(BoundingBox):
    landmarks: Optional[List[Tuple[float,float]]] = None # (x, y) 6点
    _ear: Optional[EARValue] = None

    @property
    def ear(self) -> EARValue:
        assert self.landmarks
        assert len(self.landmarks) == 6
        if self._ear != None: return self._ear
        # Convert tuples to numpy arrays for calculation
        points = [np.array(pt) for pt in self.landmarks]
        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        # p1, p4: horizontal endpoints, p2,p6 and p3,p5: vertical pairs
        p1, p2, p3, p4, p5, p6 = points
        # 垂直距離
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5) 
        # 水平距離  
        horizontal = np.linalg.norm(p1 - p4)
        
        if horizontal == 0: ear = 0.0
        else: ear = (vertical1 + vertical2) / (2.0 * horizontal)
        self._ear = cast(EARValue, float(ear))
        return self._ear


@dataclass
class FaceEyeBoundingBox(FaceBoundingBox):
    left_eye: Optional[Union[EyeBoundingBox, BoundingBox]] = None
    right_eye: Optional[Union[EyeBoundingBox, BoundingBox]] = None
    landmarks: Optional[FloatArray] = None  # 顔全体のランドマーク