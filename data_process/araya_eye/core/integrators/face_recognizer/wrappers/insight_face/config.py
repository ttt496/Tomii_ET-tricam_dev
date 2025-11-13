from typing import Literal
from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass

from core.models.insight_face import InsightFaceConfig

@pydantic_dataclass
@dataclass
class Optimization:
    enabled: bool                           # 最適化処理の有効/無効
    position_similarity_threshold: float    # 位置類似度閾値（IoU）
    image_similarity_threshold: float       # 画像差分類似度閾値
    skip_interval_seconds: float            # 定期的な顔ベクトル作成間隔（秒）
    max_skip_frames: int                    # 最大省略フレーム数（30fps想定で10秒）
    position_weight: float                  # 位置情報の重み（0.0-1.0）
    image_comparison_method: Literal["mse", "ssim", "histogram"] # 画像比較手法

@pydantic_dataclass
@dataclass
class InsightFaceRecognizerConfig:
    config: InsightFaceConfig
    similarity_threshold: float
    max_persons: int
    optimization: Optimization