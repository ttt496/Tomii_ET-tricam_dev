# train_tri_cam_v2.py
# -*- coding: utf-8 -*-
"""
最小学習ループ（デモ用）
- Tri-Cam v2 モデルの学習・検証の骨組み。
- 実データがない環境でもダミーデータで動作を確認可能。

使い方
------
# ダミーデータでワンエポック回す:
python train_tri_cam_v2.py --dummy 1 --epochs 1

# 実データのテンソル(npz)を与える:
#  npz には以下の名前で格納:
#   - eye_patches: (N, 2*n_cams, C, H, W)
#   - eye_coords : (N, 2*n_cams, 3)
#   - gaze       : (N, 2)       # [0,1] or pixel, モデル出力とスケールを揃える
python train_tri_cam_v2.py --data /path/to/data.npz --epochs 50 --batch 64
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from training.tri_cam_model import TriCamConfig, TriCamNet, compute_loss


class TriCamTensorDataset(Dataset):
    def __init__(self, eye_patches: np.ndarray, eye_coords: np.ndarray, gaze: np.ndarray):
        assert eye_patches.ndim == 5 and eye_coords.ndim == 3 and gaze.ndim == 2
        self.eye_patches = torch.from_numpy(eye_patches).float()
        self.eye_coords  = torch.from_numpy(eye_coords).float()
        self.gaze        = torch.from_numpy(gaze).float()

    def __len__(self): return self.gaze.shape[0]

    def __getitem__(self, idx):
        return self.eye_patches[idx], self.eye_coords[idx], self.gaze[idx]


def make_dummy(n: int = 256, n_cams: int = 3, C: int = 1, H: int = 20, W: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = n_cams * 2
    eye_patches = np.random.randn(n, E, C, H, W).astype(np.float32)
    eye_coords  = np.random.rand(n, E, 3).astype(np.float32)
    # miss をやや多めに
    miss_mask = (np.random.rand(n, E) > 0.85).astype(np.float32)
    eye_coords[..., 2] = miss_mask
    gaze = np.random.rand(n, 2).astype(np.float32)
    return eye_patches, eye_coords, gaze


def train_one_epoch(
    model: TriCamNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_weight: float,
):
    model.train()
    total = 0.0
    for eye_p, eye_c, gaze in loader:
        eye_p, eye_c, gaze = eye_p.to(device), eye_c.to(device), gaze.to(device)
        out = model(eye_p, eye_c)
        loss, logs = compute_loss(out, gaze, eye_c, aux_weight=aux_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(logs["loss"]) * eye_p.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: TriCamNet,
    loader: DataLoader,
    device: torch.device,
    aux_weight: float,
):
    model.eval()
    total = 0.0
    for eye_p, eye_c, gaze in loader:
        eye_p, eye_c, gaze = eye_p.to(device), eye_c.to(device), gaze.to(device)
        out = model(eye_p, eye_c)
        loss, logs = compute_loss(out, gaze, eye_c, aux_weight=aux_weight)
        total += float(logs["loss"]) * eye_p.size(0)
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="npz path with eye_patches, eye_coords, gaze")
    ap.add_argument("--dummy", type=int, default=0, help="use dummy data")
    ap.add_argument("--n_cams", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--screen_w", type=int, default=0)
    ap.add_argument("--screen_h", type=int, default=0)
    args = ap.parse_args()

    if args.data is None and args.dummy == 0:
        raise SystemExit("Provide --data npz or set --dummy 1")

    if args.data:
        d = np.load(args.data)
        eye_patches = d["eye_patches"]
        eye_coords  = d["eye_coords"]
        gaze        = d["gaze"]
        n_cams = eye_patches.shape[1] // 2
    else:
        eye_patches, eye_coords, gaze = make_dummy(n=512, n_cams=args.n_cams)
        n_cams = args.n_cams

    ds = TriCamTensorDataset(eye_patches, eye_coords, gaze)
    n_train = int(len(ds) * 0.8)
    n_val   = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    screen_wh = None
    if args.screen_w > 0 and args.screen_h > 0:
        screen_wh = (args.screen_w, args.screen_h)

    cfg = TriCamConfig(n_cams=n_cams, in_ch=eye_patches.shape[2], patch_h=eye_patches.shape[3],
                       patch_w=eye_patches.shape[4], screen_wh=screen_wh)
    model = TriCamNet(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, torch.device(args.device), cfg.aux_weight)
        va = eval_epoch(model, val_loader, torch.device(args.device), cfg.aux_weight)
        dt = time.time() - t0
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "tri_cam_v2_best.pt")
        print(f"[{ep:03d}] train {tr:.6f} | val {va:.6f} | best {best:.6f} | {dt:.1f}s")

    print("Done. Best checkpoint: tri_cam_v2_best.pt")


if __name__ == "__main__":
    main()
