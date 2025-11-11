from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from radam import RAdam
except ImportError:  # simple fallback
    RAdam = optim.Adam

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

import sys
from pathlib import Path as _Path
THIS_DIR = _Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset.eye_dataset import (
    SampleRecord,
    TriCamEyeDataset,
    load_eye_crop_samples,
    split_by_session,
)
from training.model.tri_cam_model import TriCamConfig, TriCamNet
from training.loss.tricam_loss import TriCamLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_sessions(base_dir: Path, manifest_name: str) -> List[str]:
    sessions: List[str] = []
    for session_dir in sorted(base_dir.rglob("*")):
        if not session_dir.is_dir():
            continue
        manifest_path = session_dir / manifest_name
        if not manifest_path.exists():
            continue
        try:
            rel = session_dir.relative_to(base_dir)
        except ValueError:
            rel = session_dir
        sessions.append(str(rel))
    return sessions


def prepare_datasets(cfg: DictConfig) -> Tuple[TriCamEyeDataset, TriCamEyeDataset, TriCamEyeDataset]:
    data_cfg = cfg.data
    base_dir = Path(data_cfg.base_dir).expanduser().resolve()
    manifest_name = data_cfg.get("manifest_name", "eye_crops.jsonl")
    allowed_phases = data_cfg.get("allowed_phases")

    train_sessions = set(data_cfg.train_sessions or [])
    val_sessions = set(data_cfg.val_sessions or [])
    test_sessions = set(data_cfg.test_sessions or [])
    specified_sessions = train_sessions | val_sessions | test_sessions

    if specified_sessions:
        sessions_to_load = sorted(specified_sessions)
    else:
        sessions_to_load = discover_sessions(base_dir, manifest_name)

    session_samples: Dict[str, List[SampleRecord]] = {}
    for session in sessions_to_load:
        session_path = (base_dir / session).resolve()
        manifest_path = session_path / manifest_name
        if not manifest_path.exists():
            manifests = list(session_path.glob("*.jsonl"))
            if manifests:
                manifest_path = manifests[0]
        if not manifest_path.exists():
            print(f"Warning: manifest not found for session {session}: {session_path}")
            continue
        samples = load_eye_crop_samples(manifest_path, base_dir=session_path, allowed_phases=allowed_phases)
        if samples:
            session_samples[session] = samples

    if not session_samples:
        raise RuntimeError("No samples found for any session.")

    all_samples = [s for samples in session_samples.values() for s in samples]

    camera_ids = list(data_cfg.camera_ids) if data_cfg.get("camera_ids") else sorted(
        set(rec.camera_id for sample in all_samples for rec in sample.records.values())
    )
    if len(camera_ids) != data_cfg.n_cams:
        raise ValueError(f"Expected {data_cfg.n_cams} cameras but found {camera_ids}")

    if specified_sessions:
        def collect(session_list: Iterable[str]) -> List[SampleRecord]:
            out: List[SampleRecord] = []
            for session in session_list:
                out.extend(session_samples.get(session, []))
            return out

        train_samples = collect(train_sessions)
        val_samples = collect(val_sessions)
        test_samples = collect(test_sessions)
    else:
        train_samples, val_samples, test_samples = split_by_session(
            all_samples,
            val_ratio=data_cfg.val_ratio,
            test_ratio=data_cfg.test_ratio,
            seed=cfg.seed,
            group_mode=data_cfg.split_group,
        )

    patch_size = (cfg.model.patch_h, cfg.model.patch_w)
    frame_size = (data_cfg.frame_width, data_cfg.frame_height)

    ds_kwargs = dict(
        camera_ids=camera_ids,
        frame_size=frame_size,
        patch_size=patch_size,
        image_format=data_cfg.image_format,
    )
    train_ds = TriCamEyeDataset(train_samples, **ds_kwargs)
    val_ds = TriCamEyeDataset(val_samples, **ds_kwargs)
    test_ds = TriCamEyeDataset(test_samples, **ds_kwargs)
    return train_ds, val_ds, test_ds


def build_dataloaders(
    train_ds: TriCamEyeDataset,
    val_ds: TriCamEyeDataset,
    test_ds: TriCamEyeDataset,
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def init_wandb(cfg: DictConfig, run_name: str) -> Any:
    if cfg.wandb.mode == "disabled" or wandb is None:
        return None
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return wandb


def train_one_epoch(
    loader: DataLoader,
    model: TriCamNet,
    criterion: TriCamLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_cams: int,
    epoch: int,
    grad_clip: float,
    log: Any,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_batches = 0
    dist_accum = 0.0
    sample_count = 0
    dist_accum = 0.0
    sample_count = 0
    for batch in loader:
        eye_patches = batch["eye_patches"].to(device)
        eye_coords = batch["eye_coords"].to(device)
        gaze = batch["gaze"].to(device)
        aux_target = batch["aux_target"].to(device)
        valid_eye = batch["valid_eye"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(eye_patches, eye_coords)
        aux_pred = (
            outputs["aux"]
            .view(eye_patches.size(0), n_cams, 2, 2)
            .permute(0, 2, 1, 3)
        )  # (B,2,n_cams,2)
        loss_dict = criterion(
            gaze_pred=outputs["gaze"],
            gaze_gt=gaze,
            pred_aux_xy=aux_pred,
            gt_eye_xy=aux_target,
            valid_eye=valid_eye,
        )
        loss = loss_dict["total"]
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_main += loss_dict["main"].item()
        total_aux += loss_dict["aux"].item()
        total_batches += 1
        with torch.no_grad():
            diff = outputs["gaze"] - gaze
            dist_accum += torch.sum(torch.sqrt(torch.sum(diff ** 2, dim=-1))).item()
            sample_count += gaze.size(0)

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "loss_main": total_main / max(total_batches, 1),
        "loss_aux": total_aux / max(total_batches, 1),
        "gaze_l2": dist_accum / max(sample_count, 1),
    }
    if log:
        log.log({f"train/{k}": v for k, v in metrics.items()}, step=epoch)
    return metrics


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: TriCamNet,
    criterion: TriCamLoss,
    device: torch.device,
    n_cams: int,
    split: str,
    epoch: int,
    log: Any,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_batches = 0
    dist_accum = 0.0
    sample_count = 0

    for batch in loader:
        eye_patches = batch["eye_patches"].to(device)
        eye_coords = batch["eye_coords"].to(device)
        gaze = batch["gaze"].to(device)
        aux_target = batch["aux_target"].to(device)
        valid_eye = batch["valid_eye"].to(device)

        outputs = model(eye_patches, eye_coords)
        aux_pred = (
            outputs["aux"]
            .view(eye_patches.size(0), n_cams, 2, 2)
            .permute(0, 2, 1, 3)
        )

        loss_dict = criterion(
            gaze_pred=outputs["gaze"],
            gaze_gt=gaze,
            pred_aux_xy=aux_pred,
            gt_eye_xy=aux_target,
            valid_eye=valid_eye,
        )
        total_loss += loss_dict["total"].item()
        total_main += loss_dict["main"].item()
        total_aux += loss_dict["aux"].item()
        total_batches += 1
        diff = outputs["gaze"] - gaze
        dist_accum += torch.sum(torch.sqrt(torch.sum(diff ** 2, dim=-1))).item()
        sample_count += gaze.size(0)

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "loss_main": total_main / max(total_batches, 1),
        "loss_aux": total_aux / max(total_batches, 1),
        "gaze_l2": dist_accum / max(sample_count, 1),
    }
    if log:
        log.log({f"{split}/{k}": v for k, v in metrics.items()}, step=epoch)
    return metrics


@hydra.main(version_base=None, config_path="config", config_name="train_tri_cam")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = prepare_datasets(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(train_ds, val_ds, test_ds, cfg)
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model_cfg = TriCamConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = TriCamNet(model_cfg).to(device)

    criterion = TriCamLoss(
        main=cfg.loss.main,
        aux_weight=cfg.loss.aux_weight,
        huber_delta=cfg.loss.huber_delta,
    )
    if cfg.optimizer.name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "radamschedulefree":
        optimizer = RAdam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {cfg.optimizer.name}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    run_name = cfg.wandb.run_name or f"tri_cam_{cfg.seed}"
    log = init_wandb(cfg, run_name)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.training.max_epochs + 1):
        train_metrics = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            cfg.data.n_cams,
            epoch,
            cfg.training.grad_clip,
            log,
        )
        val_metrics = evaluate(
            val_loader,
            model,
            criterion,
            device,
            cfg.data.n_cams,
            "val",
            epoch,
            log,
        )
        scheduler.step(val_metrics["loss"])
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_l2={train_metrics['gaze_l2']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_l2={val_metrics['gaze_l2']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }

    if best_state:
        ckpt_path = Path("best_model.pt")
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint to {ckpt_path.resolve()}")

    test_metrics = evaluate(
        test_loader,
        model,
        criterion,
        device,
        cfg.data.n_cams,
        "test",
        cfg.training.max_epochs,
        log,
    )
    print(f"Test metrics: {test_metrics}")
    print(f"Best val gaze L2: {best_val:.4f}, test gaze L2: {test_metrics.get('gaze_l2', 0.0):.4f}")

    if log:
        log.finish()


if __name__ == "__main__":
    main()
def _gaze_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    diff = pred - target
    mse = torch.mean((diff) ** 2).item()
    l2 = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=-1))).item()
    return {"mse": mse, "l2": l2}
