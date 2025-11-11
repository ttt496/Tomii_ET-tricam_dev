from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
IntArray = NDArray[np.int32]

Metadata = Dict[str, Any]  # メタデータ

Timestamp = float  # Unix timestamp
FrameID = int
Duration = float  # 秒