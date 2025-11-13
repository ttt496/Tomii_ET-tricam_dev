from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

Confidence = float  # 0.0 - 1.0
Coordinate = Tuple[float, float]  # (x, y)

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    confidence: Confidence = 1.0
    
    @property
    def x1(self) -> float:
        return self.x
    
    @property
    def y1(self) -> float:
        return self.y
    
    @property
    def x2(self) -> float:
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        return self.y + self.height
    
    @property
    def center(self) -> Coordinate:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: BoundingBox) -> float:
        """IoU (Intersection over Union) を計算"""
        # 交差領域の計算
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        # 交差がない場合
        if x2 <= x1 or y2 <= y1: return 0.0
        
        # 交差面積と合計面積の計算
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0