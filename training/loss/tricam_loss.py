
# tri_cam_loss.py
# -----------------------------------------------------------------------------
# Losses for Tri‑Cam style gaze estimation training.
#
# Features
# --------
# 1) Main gaze loss:
#    - 'mse': mean squared error on (x, y)
#    - 'huber': smooth L1 (robust) on (x, y)
#    - 'hetero': heteroscedastic regression
#       model outputs (x, y, s) where s = log(sigma^2).
#       L = ||g - g_hat||^2 / exp(s) + s
#
# 2) Intra-validation auxiliary loss:
#    - Per-eye, per-camera masked-eye coordinate regression.
#      Shape convention:
#        gt_eye_xy:     (B, E=2, C=3, 2)  # ground truth eye centers (pixels or normalized)
#        pred_aux_xy:   (B, E=2, C=3, 2)  # predictions for the "masked camera" coords
#        valid_eye:     (B, E=2, C=3)     # bool; True when that eye is detected on that camera
#    - Missing eyes (coords == (-1, -1)) are automatically masked.
#
# 3) Aggregation:
#    total = main + aux_weight * aux
#
# Usage (example)
# ---------------
#   from tri_cam_loss import TriCamLoss
#   crit = TriCamLoss(main='mse', aux_weight=0.1, huber_delta=2.0)
#   out = crit(gaze_pred, gaze_gt, aux_pred, eye_gt)
#   total = out['total']; aux = out['aux']; main = out['main']
#
# Notes
# -----
# - Units: you can train in pixels or normalized coordinates; be consistent across
#   predictions and ground truths.
# - If you use heteroscedastic mode, supply gaze_pred as (..., 3) = (x,y,logvar).
# - If you do not predict auxiliary heads, pass pred_aux_xy=None to skip aux loss.
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TriCamLossConfig:
    main: str = "mse"            # 'mse' | 'huber' | 'hetero'
    huber_delta: float = 2.0     # for 'huber'
    aux_weight: float = 0.1      # intra-validation weight (paper uses 0.1)
    reduction: str = "mean"      # 'mean' | 'sum'


class TriCamLoss(nn.Module):
    def __init__(self, main: str = "mse", aux_weight: float = 0.1, huber_delta: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.cfg = TriCamLossConfig(main=main, huber_delta=huber_delta, aux_weight=aux_weight, reduction=reduction)
        if self.cfg.main not in ("mse", "huber", "hetero"):
            raise ValueError("main must be 'mse' | 'huber' | 'hetero'")

    def forward(
        self,
        gaze_pred: torch.Tensor,          # (B, 2) for 'mse'/'huber', or (B, 3) for 'hetero' -> (x,y,logvar)
        gaze_gt: torch.Tensor,            # (B, 2)
        pred_aux_xy: Optional[torch.Tensor] = None,  # (B, 2, 3, 2)  [E=2, C=3] or None to skip
        gt_eye_xy: Optional[torch.Tensor] = None,    # (B, 2, 3, 2)
        valid_eye: Optional[torch.Tensor] = None,    # (B, 2, 3) bool
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
          - total: scalar
          - main: scalar
          - aux:  scalar (0 if aux not provided)
        """
        main = self._gaze_loss(gaze_pred, gaze_gt)

        aux = torch.zeros_like(main)
        if pred_aux_xy is not None and gt_eye_xy is not None:
            aux = self._aux_loss(pred_aux_xy, gt_eye_xy, valid_eye)

        total = main + self.cfg.aux_weight * aux
        return {"total": total, "main": main, "aux": aux}

    # ------------------------------ main losses ------------------------------
    def _gaze_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.cfg.main == "hetero":
            # pred: (B, 3) -> x, y, s (s = log(sigma^2))
            if pred.shape[-1] != 3:
                raise ValueError("hetero expects pred shape (B,3): (x,y,logvar)")
            g_hat = pred[..., :2]
            s = pred[..., 2:3]  # (B,1)
            sq = (gt - g_hat) ** 2  # (B,2)
            # Loss per-sample
            L = (sq.sum(dim=-1, keepdim=True) / torch.exp(s) + s)  # (B,1)
            return self._reduce(L)
        elif self.cfg.main == "huber":
            if pred.shape[-1] != 2:
                raise ValueError("huber expects pred shape (B,2)")
            L = F.huber_loss(pred, gt, delta=self.cfg.huber_delta, reduction="none").sum(dim=-1, keepdim=True)
            return self._reduce(L)
        else:  # mse
            if pred.shape[-1] != 2:
                raise ValueError("mse expects pred shape (B,2)")
            L = ((pred - gt) ** 2).sum(dim=-1, keepdim=True)
            return self._reduce(L)

    # ------------------------- intra-validation aux --------------------------
    def _aux_loss(self, pred_aux: torch.Tensor, gt_eye: torch.Tensor, valid_eye: Optional[torch.Tensor]) -> torch.Tensor:
        """
        pred_aux: (B, 2, 3, 2) predicted masked-eye coordinates for each eye & camera
        gt_eye:   (B, 2, 3, 2) ground truth eye centers (coords or normalized)
        valid_eye:(B, 2, 3)    bool; when False the target is missing and must be masked out

        Returns reduced scalar.
        """
        if pred_aux.shape != gt_eye.shape:
            raise ValueError(f"pred_aux and gt_eye shape mismatch: {pred_aux.shape} vs {gt_eye.shape}")

        # Mask missing targets: coords == (-1, -1) or valid_eye == False
        with torch.no_grad():
            mask_coords = (gt_eye[..., 0] >= 0) & (gt_eye[..., 1] >= 0)  # (B,2,3)
            if valid_eye is not None:
                mask = mask_coords & valid_eye.bool()
            else:
                mask = mask_coords
            mask = mask.unsqueeze(-1)  # (B,2,3,1)

        diff = pred_aux - gt_eye                       # (B,2,3,2)
        sq = (diff ** 2) * mask                        # mask invalid
        # sum over xy
        per = sq.sum(dim=-1)                           # (B,2,3)
        # avoid empty: if all masked, return 0
        if mask.sum() == 0:
            return torch.zeros((), device=pred_aux.device, dtype=pred_aux.dtype)

        # reduce
        if self.cfg.reduction == "sum":
            return per.sum() / (mask.sum() + 1e-9)
        else:
            # mean over valid terms
            return per.sum() / (mask.sum() + 1e-9)

    # ------------------------------- utilities --------------------------------
    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1)
        if self.cfg.reduction == "sum":
            return x.sum() / x.shape[0]
        return x.mean()
