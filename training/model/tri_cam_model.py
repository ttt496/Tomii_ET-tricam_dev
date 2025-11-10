# tri_cam_model_v2.py
# -*- coding: utf-8 -*-
"""
Tri-Cam model (再実装 v2, PyTorch)

主な改良点（v1→v2）
-------------------
- **可変カメラ数**に対応（既定は3）。内部は (n_cams * 2) 眼で一般化。
- **品質重み付けの安定化**：全眼欠損時に softmax が NaN にならないフォールバックを実装。
- **スクリーン座標スケーリング**：gaze を [0,1] 正規化出力／ピクセル出力のどちらにも対応。
- **EMA 平滑**や **ONNX エクスポート**の補助機能を付属。
- Docstring/TypeHint を充実。学習・推論の足場としてすぐ使える。

前提
----
- 眼パッチ: (B, 2 * n_cams, C, H, W)  例: n_cams=3 → 6枚 (C=1推奨, H=20, W=40)
- 眼座標:   (B, 2 * n_cams, 3) = (u, v, miss)  # u,vは0-1正規化, miss∈{0,1}
- 入力順序: [cam0-L, cam0-R, cam1-L, cam1-R, ..., cam{n-1}-L, cam{n-1}-R]

出力
----
- gaze: (B,2)  # 既定: 0-1 正規化。cfg.screen_wh が与えられた場合はピクセルで返す。
- attn: (B, 2*n_cams)  # パッチ重み（softmax後）
- aux:  (B, n_cams, 4) # (uL,vL,uR,vR) の補助予測
- img_feat, geo_feat: デバッグ用

著者注
------
- 本コードは論文本体からの再現実装であり、学習済み重みは含みません。
- 細部（チャネル数/層幅等）は実用性重視の設計判断を含みます。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Config
# =========================
@dataclass
class TriCamConfig:
    n_cams: int = 3
    eyes_per_cam: int = 2         # 固定で2（L/R）
    in_ch: int = 1                # 眼パッチのチャネル（1=Grayが推奨）
    patch_h: int = 20
    patch_w: int = 40
    eye_feat_dim: int = 96
    geo_hidden: int = 192
    fusion_hidden: int = 128
    dropout: float = 0.1
    aux_weight: float = 0.1
    # 出力座標スケール: None -> [0,1]、それ以外 -> ピクセル出力へスケール
    screen_wh: Optional[Tuple[int, int]] = None  # (W,H) を与えるとピクセル座標で出力


# =========================
# Building Blocks
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class EyeEncoder(nn.Module):
    """共有CNN: 眼パッチを特徴ベクトルへ"""
    def __init__(self, in_ch: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_ch, 16, stride=2),  # (H/2, W/2)
            ConvBlock(16, 32, stride=2),     # (H/4, W/4)
            ConvBlock(32, 64, stride=2),     # (H/8, W/8)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # 品質スコア
        self.q_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W)
        returns:
            feat: (B, out_dim)
            q:    (B, 1)  # softmax前のスコア
        """
        h = self.backbone(x)
        feat = self.head(h)
        q = self.q_head(feat)
        return feat, q


class GeoEncoder(nn.Module):
    """幾何MLP: (u,v,miss)x(2*n_cams) → 埋め込み"""
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Tri-Cam Model
# =========================
class TriCamNet(nn.Module):
    """
    Tri-Cam reproduction model (v2).

    入力順序は [cam0-L, cam0-R, cam1-L, cam1-R, ..., camN-1-R] を前提。
    eye_coords の各要素は (u, v, miss) で、u,v∈[0,1], miss∈{0,1} を想定。
    """
    def __init__(self, cfg: TriCamConfig):
        super().__init__()
        self.cfg = cfg
        self.n_eyes = cfg.n_cams * cfg.eyes_per_cam  # 2*n_cams

        # 画像分岐（共有CNN）
        self.eye_enc = EyeEncoder(cfg.in_ch, cfg.eye_feat_dim, dropout=cfg.dropout)

        # 幾何分岐
        geo_in_dim = self.n_eyes * 3  # (u,v,miss) x (2*n_cams)
        self.geo_enc = GeoEncoder(geo_in_dim, cfg.geo_hidden, dropout=cfg.dropout)

        # gaze 回帰ヘッド（融合）
        self.fuse = nn.Sequential(
            nn.Linear(cfg.geo_hidden + cfg.eye_feat_dim, cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, 2),  # (x,y) in [0,1]
        )

        # 補助出力（各カメラの (uL, vL, uR, vR) を予測）
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.geo_hidden, cfg.geo_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.geo_hidden // 2, 4),
            ) for _ in range(cfg.n_cams)
        ])

    # ---------- utils ----------
    def cam_eye_indices(self) -> List[Tuple[int, int]]:
        """各カメラの (L_idx, R_idx) を返す。"""
        idx = []
        for c in range(self.cfg.n_cams):
            li = 2 * c
            ri = 2 * c + 1
            idx.append((li, ri))
        return idx

    def mask_camera_in_coords(self, eye_coords: torch.Tensor, cam_idx: int) -> torch.Tensor:
        """
        指定カメラの (u,v) を 0、miss=1 にしてマスク。
        eye_coords: (B, 2*n_cams, 3)
        """
        masked = eye_coords.clone()
        li, ri = self.cam_eye_indices()[cam_idx]
        masked[:, [li, ri], :2] = 0.0
        masked[:, [li, ri],  2] = 1.0
        return masked

    # ---------- forward ----------
    def forward(
        self,
        eye_patches: torch.Tensor,   # (B, 2*n_cams, C, H, W)
        eye_coords: torch.Tensor,    # (B, 2*n_cams, 3) -> (u,v,miss)
        ) -> Dict[str, torch.Tensor]:
        B, n2, C, H, W = eye_patches.shape
        assert n2 == self.n_eyes, f"2*n_cams={self.n_eyes} に一致していません"
        assert eye_coords.shape[:2] == (B, self.n_eyes)

        # ---- 画像分岐 ----
        patches = eye_patches.reshape(B * self.n_eyes, C, H, W)
        feats, qs = self.eye_enc(patches)               # feats: (B*E, F), qs: (B*E, 1)
        feats = feats.view(B, self.n_eyes, -1)          # (B,E,F)
        qs = qs.view(B, self.n_eyes)                    # (B,E)

        # 欠損をsoftmax前に強制マスク
        miss = (eye_coords[..., 2] > 0.5)               # (B,E) True=欠損
        qs_masked = qs.masked_fill(miss, -1e9)

        # softmax（全欠損フォールバック: 一様重み）
        attn = F.softmax(qs_masked, dim=1)              # (B,E)
        all_missing = (miss.sum(dim=1) == self.n_eyes)  # (B,)
        if all_missing.any():
            uniform = torch.full_like(attn, 1.0 / self.n_eyes)
            attn = torch.where(all_missing.unsqueeze(1), uniform, attn)

        # 加重和特徴
        f_img = torch.einsum('be,bef->bf', attn, feats) # (B,F)

        # ---- 幾何分岐 ----
        geo_in = eye_coords.reshape(B, -1)              # (B, E*3)
        g = self.geo_enc(geo_in)                        # (B, geo_hidden)

        # ---- gaze 回帰 ----
        fused = torch.cat([g, f_img], dim=1)            # (B, geo_hidden + F)
        gaze_norm = self.fuse(fused).sigmoid()          # (B,2) in [0,1]

        # スケール（ピクセル出力が望まれる場合）
        if self.cfg.screen_wh is None:
            gaze = gaze_norm
        else:
            Wscr, Hscr = self.cfg.screen_wh
            scale = torch.tensor([Wscr, Hscr], dtype=gaze_norm.dtype, device=gaze_norm.device)
            gaze = gaze_norm * scale

        # ---- 補助出力 (n_cams) ----
        aux_preds = []
        for k in range(self.cfg.n_cams):
            masked_coords = self.mask_camera_in_coords(eye_coords, k).reshape(B, -1)  # (B,E*3)
            gk = self.geo_enc(masked_coords)                                          # 共有エンコーダで再利用
            pk = self.aux_heads[k](gk)                                               # (B,4) -> (uL,vL,uR,vR)
            aux_preds.append(pk)
        aux_preds = torch.stack(aux_preds, dim=1)  # (B, n_cams, 4)

        return {
            "gaze": gaze,              # (B,2)  [0,1] or pixel
            "gaze_norm": gaze_norm,    # (B,2)  常に[0,1]
            "attn": attn,              # (B,E)
            "img_feat": f_img,         # (B,F)
            "geo_feat": g,             # (B,geo_hidden)
            "aux": aux_preds,          # (B,n_cams,4)
        }


# =========================
# Loss / Metrics
# =========================
def mse_masked(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """要素マスク付き MSE（平均）。mask=1 が有効要素。"""
    diff2 = (pred - tgt) ** 2
    num = (diff2 * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def compute_loss(
    out: Dict[str, torch.Tensor],
    tgt_gaze: torch.Tensor,        # (B,2)  [0,1] or pixelに関わらず out["gaze"] と同スケールを渡す
    eye_coords: torch.Tensor,      # (B, 2*n_cams, 3) for aux GT
    aux_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    補助損失:
      各カメラkについて GT=(uL,vL,uR,vR) を eye_coords から取り出す。
      miss=1 の要素は損失から除外（無効化）。
    """
    gaze = out["gaze"]                     # (B,2)
    aux = out["aux"]                       # (B,n_cams,4)
    B, n_cams, _ = aux.shape

    # main
    loss_main = F.mse_loss(gaze, tgt_gaze)

    # aux
    loss_aux_total = 0.0
    valid_terms = 0.0

    for k in range(n_cams):
        li, ri = 2 * k, 2 * k + 1
        gt = torch.stack([
            eye_coords[:, li, 0], eye_coords[:, li, 1],
            eye_coords[:, ri, 0], eye_coords[:, ri, 1],
        ], dim=1)  # (B,4)

        miss_l = eye_coords[:, li, 2] > 0.5
        miss_r = eye_coords[:, ri, 2] > 0.5
        mask = torch.stack([~miss_l, ~miss_l, ~miss_r, ~miss_r], dim=1).float()  # (B,4)

        loss_k = mse_masked(aux[:, k, :], gt, mask)
        loss_aux_total = loss_aux_total + loss_k
        valid_terms = valid_terms + 1.0

    loss_aux = loss_aux_total / max(valid_terms, 1.0)

    loss = loss_main + aux_weight * loss_aux
    loss_dict = {
        "loss": loss.detach(),
        "loss_main": loss_main.detach(),
        "loss_aux": loss_aux.detach(),
    }
    return loss, loss_dict


# =========================
# Inference helper (EMA)
# =========================
class EMASmoother:
    """gaze平滑用の簡易EMA"""
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.state: Optional[torch.Tensor] = None

    def reset(self):
        self.state = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            self.state = x.detach().clone()
        else:
            self.state = self.alpha * self.state + (1 - self.alpha) * x
        return self.state


# =========================
# Export helper
# =========================
def export_onnx(
    model: TriCamNet,
    dummy_img: torch.Tensor,
    dummy_meta: torch.Tensor,
    onnx_path: str = "tri_cam.onnx",
    opset: int = 13,
):
    """ONNX エクスポート補助"""
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy_img, dummy_meta), onnx_path,
            input_names=["eye_patches", "eye_coords"],
            output_names=["gaze", "gaze_norm", "attn", "img_feat", "geo_feat", "aux"],
            dynamic_axes={"eye_patches": {0: "B"}, "eye_coords": {0: "B"},
                          "gaze": {0: "B"}, "gaze_norm": {0: "B"}, "attn": {0: "B"},
                          "img_feat": {0: "B"}, "geo_feat": {0: "B"}, "aux": {0: "B"}},
            opset_version=opset,
        )


if __name__ == "__main__":
    # 形状自己テスト
    cfg = TriCamConfig(n_cams=3, in_ch=1, patch_h=20, patch_w=40, screen_wh=None)
    model = TriCamNet(cfg)

    B = 2
    E = cfg.n_cams * 2
    x_img = torch.randn(B, E, cfg.in_ch, cfg.patch_h, cfg.patch_w)
    x_meta = torch.rand(B, E, 3)
    x_meta[...,2] = (x_meta[...,2] > 0.8).float()  # ダミー欠損
    y = torch.rand(B, 2)

    out = model(x_img, x_meta)
    loss, d = compute_loss(out, y, x_meta, aux_weight=cfg.aux_weight)

    print("gaze:", out["gaze"].shape)
    print("attn:", out["attn"].shape)
    print("aux :", out["aux"].shape)
    print("loss:", float(d["loss"]))
