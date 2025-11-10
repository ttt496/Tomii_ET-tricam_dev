from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


EyeKey = Tuple[str, int]  # (eye_side, camera_id)


@dataclass
class EyeRecord:
    image_path: Path
    image_npy_path: Optional[Path]
    center: Optional[Tuple[float, float]]
    camera_id: int
    eye_side: str
    csv_row: Dict[str, str]


@dataclass
class SampleRecord:
    session: str
    frame_index: int
    timestamp: float
    gaze: Tuple[float, float]
    records: Dict[Tuple[int, str], EyeRecord]


def load_eye_crop_samples(
    jsonl_path: Path,
    base_dir: Optional[Path] = None,
    allowed_phases: Optional[Sequence[str]] = None,
) -> List[SampleRecord]:
    """Aggregate per-eye records into per-frame samples."""
    jsonl_path = Path(jsonl_path)
    if base_dir is None:
        base_dir = jsonl_path.parent
    groups: Dict[Tuple[str, int], Dict[str, object]] = {}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            session = rec["session"]
            frame_idx = int(rec["frame_index"])
            key = (session, frame_idx)
            phase = csv_row.get("phase")

            group = groups.setdefault(
                key,
                {
                    "session": session,
                    "frame_index": frame_idx,
                    "timestamp": float(rec.get("timestamp", rec.get("csv_row", {}).get("timestamp_sec", 0.0))),
                    "gaze": None,
                    "records": {},
                    "valid_phase": True,
                },
            )
            if allowed_phases is not None:
                if phase is None:
                    group["valid_phase"] = False
                else:
                    group["valid_phase"] = group["valid_phase"] and (phase in allowed_phases)

            csv_row = rec.get("csv_row", {})
            if group["gaze"] is None:
                try:
                    gaze_x = float(csv_row["norm_x"])
                    gaze_y = float(csv_row["norm_y"])
                    group["gaze"] = (gaze_x, gaze_y)
                except (KeyError, ValueError):
                    group["gaze"] = None

            eye_side = rec["eye_side"]
            camera_id = int(rec["camera_id"])
            image_path = Path(base_dir, rec["image_path"]).resolve()
            image_npy_path = rec.get("image_npy_path")
            if image_npy_path:
                image_npy_path = Path(base_dir, image_npy_path).resolve()

            center = None
            cx = rec.get("eye_center_x")
            cy = rec.get("eye_center_y")
            if cx is not None and cy is not None:
                center = (float(cx), float(cy))

            group["records"][(camera_id, eye_side)] = EyeRecord(
                image_path=image_path,
                image_npy_path=image_npy_path,
                center=center,
                camera_id=camera_id,
                eye_side=eye_side,
                csv_row=csv_row,
            )

    samples: List[SampleRecord] = []
    for (_session, _frame), data in groups.items():
        gaze = data["gaze"]
        if gaze is None:
            continue
        if not data.get("valid_phase", True):
            continue
        samples.append(
            SampleRecord(
                session=data["session"],
                frame_index=data["frame_index"],
                timestamp=float(data["timestamp"]),
                gaze=gaze,
                records=data["records"],
            )
        )

    samples.sort(key=lambda x: (x.session, x.frame_index))
    return samples


def split_by_session(
    samples: Sequence[SampleRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    group_mode: str = "session",
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    session_map: Dict[str, List[SampleRecord]] = defaultdict(list)

    def group_key(sample: SampleRecord) -> str:
        if group_mode == "day":
            return sample.session.split("/")[0]
        return sample.session
    for sample in samples:
        session_map[sample.session].append(sample)

    session_ids = list(session_map.keys())
    random.Random(seed).shuffle(session_ids)

    n_total = len(session_ids)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = max(1, int(round(n_total * test_ratio)))
    n_train = max(1, n_total - n_val - n_test)

    train_sessions = session_ids[:n_train]
    val_sessions = session_ids[n_train:n_train + n_val]
    test_sessions = session_ids[n_train + n_val:]

    def collect(ids: Iterable[str]) -> List[SampleRecord]:
        out: List[SampleRecord] = []
        for sid in ids:
            out.extend(session_map[sid])
        return out

    return collect(train_sessions), collect(val_sessions), collect(test_sessions)


class TriCamEyeDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleRecord],
        camera_ids: Sequence[int],
        frame_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        image_format: str = "png",
    ):
        self.samples = list(samples)
        self.camera_ids = list(camera_ids)
        self.frame_width, self.frame_height = frame_size
        self.patch_h, self.patch_w = patch_size
        self.image_format = image_format
        self.n_cams = len(self.camera_ids)
        self.cam_id_to_idx = {cid: idx for idx, cid in enumerate(self.camera_ids)}
        self.eye_order: List[Tuple[int, str]] = []
        for cam_id in self.camera_ids:
            for side in ("left", "right"):
                self.eye_order.append((cam_id, side))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        n_eyes = len(self.eye_order)
        eye_patches = torch.zeros((n_eyes, 1, self.patch_h, self.patch_w), dtype=torch.float32)
        eye_coords = torch.zeros((n_eyes, 3), dtype=torch.float32)
        aux_target = torch.full((2, self.n_cams, 2), -1.0, dtype=torch.float32)
        valid_eye = torch.zeros((2, self.n_cams), dtype=torch.bool)

        for eye_idx, (cam_id, eye_side) in enumerate(self.eye_order):
            record = sample.records.get((cam_id, eye_side))
            if record is None:
                eye_coords[eye_idx, 2] = 1.0
                continue
            img = self._load_image(record)
            eye_patches[eye_idx, 0] = torch.from_numpy(img)
            if record.center:
                u = float(record.center[0]) / float(self.frame_width)
                v = float(record.center[1]) / float(self.frame_height)
                eye_coords[eye_idx, 0] = u
                eye_coords[eye_idx, 1] = v
                eye_coords[eye_idx, 2] = 0.0
                cam_idx = self.cam_id_to_idx.get(cam_id, -1)
                if cam_idx == -1:
                    continue
                side_idx = 0 if eye_side == "left" else 1
                aux_target[side_idx, cam_idx, 0] = u
                aux_target[side_idx, cam_idx, 1] = v
                valid_eye[side_idx, cam_idx] = True
            else:
                eye_coords[eye_idx, 2] = 1.0

        gaze = torch.tensor(sample.gaze, dtype=torch.float32)

        meta = {
            "session": sample.session,
            "frame_index": sample.frame_index,
            "timestamp": sample.timestamp,
        }

        return {
            "eye_patches": eye_patches,
            "eye_coords": eye_coords,
            "gaze": gaze,
            "aux_target": aux_target,
            "valid_eye": valid_eye,
            "meta": meta,
        }

    def _load_image(self, record: EyeRecord) -> np.ndarray:
        if self.image_format == "npy" and record.image_npy_path and record.image_npy_path.exists():
            arr = np.load(record.image_npy_path)
        else:
            arr = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            arr = np.zeros((self.patch_h, self.patch_w), dtype=np.uint8)
        arr = cv2.resize(arr, (self.patch_w, self.patch_h))
        if arr.ndim == 2:
            norm = arr.astype(np.float32) / 255.0
        else:
            norm = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return norm
