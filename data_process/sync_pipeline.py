"""Utilities for synchronizing multi-camera calibration recordings.

The core `TemporalSynchronizer` class can be reused in online inference code by
feeding it frames as they arrive; the offline helpers below simply stream the
recorded CSV logs through the same synchronizer to generate aligned batches.
"""

from __future__ import annotations

import csv
import heapq
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass
class FrameSample:
    """Single frame record coming from one camera stream."""

    camera_id: int
    timestamp: float
    frame_index: int
    csv_row: Dict[str, str]
    video_path: Optional[Path] = None


@dataclass
class SyncedBatch:
    """Outputs a group of frames that have been temporally aligned."""

    target_timestamp: float
    samples: Dict[int, FrameSample]
    deltas: Dict[int, float]
    max_delta: float


class TemporalSynchronizer:
    """Reusable synchronizer that pairs frames from multiple cameras by timestamp.

    The same logic can be reused for online inference by calling `submit`
    whenever a new frame arrives.
    """

    def __init__(
        self,
        camera_ids: Sequence[int],
        *,
        tolerance_sec: float = 1.0 / 30.0,
    ) -> None:
        if tolerance_sec <= 0:
            raise ValueError("tolerance_sec must be positive")
        if not camera_ids:
            raise ValueError("camera_ids must not be empty")
        self.camera_ids = tuple(sorted(camera_ids))
        self.tolerance = float(tolerance_sec)
        self._buffers: Dict[int, Deque[FrameSample]] = {
            cam_id: deque() for cam_id in self.camera_ids
        }

    def submit(self, sample: FrameSample) -> Optional[SyncedBatch]:
        if sample.camera_id not in self._buffers:
            raise KeyError(f"Camera {sample.camera_id} was not registered for synchronization.")

        self._buffers[sample.camera_id].append(sample)
        return self._attempt_emit()

    def flush(self) -> List[SyncedBatch]:
        """Drain any remaining perfectly aligned batches."""
        batches: List[SyncedBatch] = []
        while True:
            batch = self._attempt_emit(drop_if_exceeds=False)
            if batch is None:
                break
            batches.append(batch)
        return batches

    def _attempt_emit(self, *, drop_if_exceeds: bool = True) -> Optional[SyncedBatch]:
        while True:
            if any(len(buf) == 0 for buf in self._buffers.values()):
                return None

            head_times = {cam_id: buf[0].timestamp for cam_id, buf in self._buffers.items()}
            target_ts = sum(head_times.values()) / len(head_times)
            deltas = {cam_id: abs(ts - target_ts) for cam_id, ts in head_times.items()}
            max_delta = max(deltas.values())

            if max_delta <= self.tolerance:
                samples = {cam_id: self._buffers[cam_id].popleft() for cam_id in self.camera_ids}
                return SyncedBatch(
                    target_timestamp=target_ts,
                    samples=samples,
                    deltas=deltas,
                    max_delta=max_delta,
                )

            if not drop_if_exceeds:
                return None

            # Drop the frame with the earliest timestamp to allow other streams to catch up.
            earliest_cam = min(head_times, key=head_times.get)
            self._buffers[earliest_cam].popleft()


@dataclass
class CameraArtifacts:
    camera_id: int
    csv_path: Path
    video_path: Optional[Path]


def discover_camera_artifacts(session_dir: Path) -> List[CameraArtifacts]:
    """Loads camera metadata either from session_meta.json or file patterns."""
    session_dir = session_dir.expanduser().resolve()
    meta_path = session_dir / "session_meta.json"
    artifacts: List[CameraArtifacts] = []

    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for cam in data.get("cameras", []):
            camera_id = int(cam["index"])
            csv_path = Path(cam["timestamp_csv"]).expanduser()
            if not csv_path.is_absolute():
                csv_path = session_dir / csv_path
            video_path = Path(cam["video_path"]).expanduser()
            if not video_path.is_absolute():
                video_path = session_dir / video_path
            artifacts.append(CameraArtifacts(camera_id, csv_path, video_path))
        return artifacts

    for csv_file in sorted(session_dir.glob("camera*_timestamps.csv")):
        stem = csv_file.stem.replace("_timestamps", "")
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        camera_id = int(digits)
        video_path = _guess_video_path(session_dir, camera_id)
        artifacts.append(CameraArtifacts(camera_id, csv_file, video_path))
    return artifacts


def _guess_video_path(session_dir: Path, camera_id: int) -> Optional[Path]:
    for ext in (".avi", ".mp4", ".mov", ".mkv"):
        candidate = session_dir / f"camera{camera_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _iter_frames(artifact: CameraArtifacts) -> Iterator[FrameSample]:
    with artifact.csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield FrameSample(
                camera_id=artifact.camera_id,
                timestamp=float(row["timestamp_sec"]),
                frame_index=int(row["frame_index"]),
                csv_row=row,
                video_path=artifact.video_path,
            )


def synchronize_session(
    session_dir: Path,
    *,
    tolerance_sec: float = 1.0 / 30.0,
) -> List[SyncedBatch]:
    """Synchronize all camera streams from a recorded session directory."""
    artifacts = discover_camera_artifacts(session_dir)
    if len(artifacts) < 2:
        raise RuntimeError("Need at least two camera recordings to synchronize.")

    synchronizer = TemporalSynchronizer(
        [art.camera_id for art in artifacts],
        tolerance_sec=tolerance_sec,
    )

    iters: Dict[int, Iterator[FrameSample]] = {art.camera_id: _iter_frames(art) for art in artifacts}
    heap: List[Tuple[float, int, FrameSample]] = []

    for cam_id, iterator in iters.items():
        try:
            sample = next(iterator)
            heapq.heappush(heap, (sample.timestamp, cam_id, sample))
        except StopIteration:
            continue

    batches: List[SyncedBatch] = []
    while heap:
        _, cam_id, sample = heapq.heappop(heap)
        batch = synchronizer.submit(sample)
        if batch:
            batches.append(batch)
        iterator = iters[cam_id]
        try:
            next_sample = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (next_sample.timestamp, cam_id, next_sample))

    batches.extend(synchronizer.flush())
    return batches


def export_synced_index(
    batches: Sequence[SyncedBatch],
    output_path: Path,
) -> None:
    """Write an index with synchronized frame references for downstream processing."""
    lines: List[Dict[str, object]] = []
    for batch in batches:
        record: Dict[str, object] = {
            "target_timestamp": batch.target_timestamp,
            "max_delta": batch.max_delta,
        }
        for cam_id, sample in batch.samples.items():
            record[f"camera{cam_id}_frame_index"] = sample.frame_index
            record[f"camera{cam_id}_timestamp"] = sample.timestamp
            if sample.video_path:
                record[f"camera{cam_id}_video_path"] = str(sample.video_path)
        lines.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(lines[0].keys()) if lines else ["target_timestamp"])
        writer.writeheader()
        for row in lines:
            writer.writerow(row)


def run_sync(session_dir: Path, tolerance: float, output: Optional[Path]) -> int:
    batches = synchronize_session(session_dir, tolerance_sec=tolerance)
    if not batches:
        print("No synchronized batches were created.")
        return 0

    output_path = output or (session_dir / "synced_index.csv")
    export_synced_index(batches, output_path)
    print(f"Synced {len(batches)} batches; wrote index to {output_path}")
    return len(batches)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Synchronize recorded calibration sessions.")
    parser.add_argument("session_dir", type=Path, help="Path to a session directory (YYYYMMDD/HHMMSS).")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0 / 30.0,
        help="Maximum timestamp delta (seconds) allowed between cameras (default: one frame at 30 FPS).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for the synchronized index (default: session_dir/synced_index.csv).",
    )
    args = parser.parse_args()
    run_sync(args.session_dir, args.tolerance, args.output)


if __name__ == "__main__":
    USE_MANUAL_ARGS = False  # Toggle to True and edit the values below for quick testing.
    if USE_MANUAL_ARGS:
        SESSION_DIR = Path("path/to/session")
        TOLERANCE = 1.0 / 30.0
        OUTPUT = None  # Or Path("synced_index.csv")
        run_sync(SESSION_DIR, TOLERANCE, OUTPUT)
    else:
        main()
