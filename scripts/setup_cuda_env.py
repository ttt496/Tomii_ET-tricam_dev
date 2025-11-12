"""CUDA 12.6環境を自動設定するスクリプト（このリポジトリでのみ有効）"""

import os
import sys
from pathlib import Path

# CUDA 12.6のパス（必要に応じて調整してください）
CUDA_BASE = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
CUDNN_BIN = Path(r"C:\tools\cudnn\bin")


def setup_cuda_env():
    """CUDA 12.6のPATHを設定（このプロセスのみ有効）"""
    if not CUDA_BASE.exists():
        print(f"Warning: CUDA 12.6 not found at {CUDA_BASE}", file=sys.stderr)
        return False
    
    # 環境変数を設定
    os.environ["CUDA_PATH"] = str(CUDA_BASE)
    os.environ["CUDA_HOME"] = str(CUDA_BASE)
    
    # PATHの先頭に追加（このプロセスのみ）
    cuda_bin = str(CUDA_BASE / "bin")
    cuda_libnvvp = str(CUDA_BASE / "libnvvp")
    cudnn_bin = str(CUDNN_BIN) if CUDNN_BIN.exists() else None
    
    current_path = os.environ.get("PATH", "")
    new_paths = [cuda_bin, cuda_libnvvp]
    if cudnn_bin:
        new_paths.append(cudnn_bin)
    
    # 既に設定されていない場合のみ追加
    for path in new_paths:
        if path not in current_path:
            os.environ["PATH"] = f"{path};{os.environ.get('PATH', '')}"
    
    return True


if __name__ == "__main__":
    if setup_cuda_env():
        print("CUDA 12.6 environment configured for this process")
        print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
    else:
        print("Failed to configure CUDA 12.6 environment", file=sys.stderr)
        sys.exit(1)

