import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Literal, Optional, NamedTuple, TypeAlias, Final, get_args, cast
from .config import YoloV9Wb25Config

# https://github.com/PINTO0309/PINTO_model_zoo/tree/main/459_YOLOv9-Wholebody25

ClassName: TypeAlias = Literal[
    "body", "adult", "child", "male", "female",
    "body_with_wheelchair", "body_with_crutches", "head", "front", "right_front",
    "right_side", "right_back", "back", "left_back", "left_side",
    "left_front", "face", "eye", "nose", "mouth",
    "ear", "hand", "hand_left", "hand_right", "foot",
]
CLASS_NAMES: Final[tuple[ClassName, ...]] = cast(
    "tuple[ClassName, ...]", get_args(ClassName)
)
CLASS_ID_TO_NAME: Final[dict[int, ClassName]] = {
    i: name for i, name in enumerate(CLASS_NAMES)
}

class YoloV9Wb25Box(NamedTuple):
    x: float
    y: float 
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: ClassName
    batch_id: Optional[int] = None

class YoloV9Wb25Model:
    def __init__(self, config: YoloV9Wb25Config):
        model_path = config.model_path
        execution_provider = config.execution_provider
        self.input_size = config.input_size
        self.object_score_threshold = config.object_score_threshold
        self.model_path = model_path

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        providers = self._setup_execution_providers(execution_provider)
        
        try:
            self.session = ort.InferenceSession(str(model_file), providers=providers)
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
            print(f"INFO: YOLO model loaded: {model_file.name}")
            print(f"INFO: Input shape: {self.input_shape}, Provider: {execution_provider}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLO model: {e}")


    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        YOLO用画像前処理
        
        Args:
            image: BGR画像 [H, W, 3]
            
        Returns:
            前処理済み画像 [1, 3, H, W] float32
        """
        # リサイズ
        resized = cv2.resize(image, self.input_size)
        
        # float32変換、CHW形式、バッチ次元追加
        processed = resized.astype(np.float32)
        transposed = np.transpose(processed, (2, 0, 1))  # HWC → CHW
        contiguous = np.ascontiguousarray(transposed, dtype=np.float32)
        batched = np.expand_dims(contiguous, axis=0)  # バッチ次元追加
        return batched
    
    def __call__(self, image: np.ndarray) -> List[YoloV9Wb25Box]:
        try:
            preprocessed_input = self._preprocess(image)
            self._validate_input(preprocessed_input)
            raw_outputs = self.session.run(None, {self.input_name: preprocessed_input})
            # ONNXのOutputをnp.ndarrayのリストにキャスト（型安全性確保）
            from typing import cast
            numpy_outputs = cast(List[np.ndarray], raw_outputs)
            boxes = self._postprocess(numpy_outputs, (image.shape[0], image.shape[1]), self.input_size)
            return boxes
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {e}")
    
    def _postprocess(self, 
                     outputs: List[np.ndarray], 
                     original_shape: Tuple[int, int], 
                     input_size: Tuple[int, int]) -> List[YoloV9Wb25Box]:
        boxes:List[YoloV9Wb25Box] = []
        if not outputs or len(outputs) == 0: return boxes
        
        try:
            raw_boxes = outputs[0]
            if raw_boxes.size == 0: return boxes
            
            # 元画像サイズと入力サイズ
            orig_h, orig_w = original_shape
            input_w, input_h = input_size
            
            # バッチ次元削除
            if len(raw_boxes.shape) == 3:
                raw_boxes = raw_boxes[0]
            
            # Post-processed vs Raw format判定
            is_post_processed = "post" in self.model_path.lower()
            
            for detection in raw_boxes:
                if len(detection) < 6: continue
                try:
                    if is_post_processed:
                        # Post-processed format: [batch_id, class_id, confidence, x1, y1, x2, y2, ...]
                        # 参考実装と完全一致の座標変換ロジック
                        batch_id = int(detection[0]) if len(detection) > 0 else None
                        class_id = int(detection[1])
                        confidence = float(detection[2])
                        x1, y1, x2, y2 = detection[3:7]
                        
                        # 参考実装と完全同一の座標変換（スケーリング計算）
                        x_min = int(max(0, x1) * orig_w / input_w)
                        y_min = int(max(0, y1) * orig_h / input_h)
                        x_max = int(min(x2, input_w) * orig_w / input_w)
                        y_max = int(min(y2, input_h) * orig_h / input_h)
                        
                        # x,y,w,h形式に変換
                        x = float(max(0, x_min))
                        y = float(max(0, y_min))
                        width = float(max(0, x_max - x_min))
                        height = float(max(0, y_max - y_min))
                        
                    else:
                        # Raw format: [x_center, y_center, width, height, confidence, class_scores...]
                        # 中心座標からx,y,w,h変換
                        x_center = detection[0]
                        y_center = detection[1]
                        width_raw = detection[2]
                        height_raw = detection[3]
                        confidence = float(detection[4])
                        
                        # 座標変換（中心→左上）
                        x = max(0, (x_center - width_raw / 2) * orig_w / input_w)
                        y = max(0, (y_center - height_raw / 2) * orig_h / input_h)
                        width = max(0, width_raw * orig_w / input_w)
                        height = max(0, height_raw * orig_h / input_h)
                        
                        # クラスID取得（最高スコア）
                        class_scores = detection[5:]
                        class_id = int(np.argmax(class_scores)) if len(class_scores) > 0 else 0
                        batch_id = None
                    
                    # 閾値チェック（有効なバウンディングボックスのみ）
                    if (confidence >= self.object_score_threshold and 
                        width > 1 and height > 1 and  # 最小サイズチェック
                        x >= 0 and y >= 0 and 
                        x + width <= orig_w and y + height <= orig_h):  # 境界チェック
                        boxes.append(YoloV9Wb25Box(
                            x=x, y=y, width=width, height=height,
                            confidence=confidence, batch_id=batch_id,
                            class_id=class_id, class_name=CLASS_ID_TO_NAME[class_id]
                        ))
                
                except (ValueError, IndexError) as e:
                    continue  # 不正なデータをスキップ
        
        except Exception as e:
            print(f"WARNING: YOLO postprocess error: {e}")
            return []
        
        return boxes
    
    def _setup_execution_providers(self, execution_provider: str) -> List[str]:
        """統一実行プロバイダー設定（優先順位管理）"""
        providers = []
        if execution_provider == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    
    def _validate_input(self, input_data: np.ndarray) -> None:
        if input_data is None: 
            raise ValueError("Input data cannot be None")
        if len(input_data.shape) != 4:
            raise ValueError(f"Expected 4D input [batch, channels, height, width], got {input_data.shape}")
        if input_data.dtype != np.float32:
            raise ValueError(f"Expected float32 input, got {input_data.dtype}")
        # バッチサイズ検証（通常は1）
        if input_data.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {input_data.shape[0]}")

