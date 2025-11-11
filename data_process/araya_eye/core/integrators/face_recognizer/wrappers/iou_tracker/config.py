from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
@dataclass
class IouTrackerConfig:
    iou_threshold: float  # IoU重複判定閾値
    max_lost_frames: int  # 追跡失敗許容フレーム数
    min_track_length: int  # 有効track最小長
    max_persons: int  # 最大人物数
    confidence_threshold: float  # 顔検出信頼度閾値
    merge_threshold: float  # track統合IoU閾値
    track_length_weight: float  # Track長重み (0.0-1.0)
    distance_penalty_factor: float  # 中心距離ペナルティ係数