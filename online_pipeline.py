from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from data_process.araya_eye.core.dtypes.media import ImageArray
from data_process.araya_eye.core.pipeline import EyeCropPipeline, PipelineFrameResult
from data_process.sync_inference import load_pipeline_config_from_yaml
from data_process.sync_pipeline import FrameSample, SyncedBatch, TemporalSynchronizer
from training.model.tri_cam_model import EMASmoother, TriCamConfig, TriCamNet

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("online_pipeline")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CameraStreamSpec:
    """Describes a live stream feeding the pipeline."""

    camera_id: int
    kind: str  # camera | video
    source: str  # device index or file path


@dataclass
class LiveFrameSample(FrameSample):
    """FrameSample with the actual image attached for online processing."""

    image: ImageArray
    frame_size: Tuple[int, int]


@dataclass
class TriCamSample:
    """Prepared eye patches + metadata for TriCam inference."""

    timestamp: float
    eye_patches: np.ndarray  # shape: (1, 2*n_cams, 1, H, W)
    eye_coords: np.ndarray  # shape: (1, 2*n_cams, 3)
    meta: Dict[str, object]


@dataclass
class TriCamOutput:
    """Final gaze estimation result."""

    timestamp: float
    gaze: Tuple[float, float]
    attn: Sequence[float]
    meta: Dict[str, object]

    def to_json(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "gaze": {"x": self.gaze[0], "y": self.gaze[1]},
            "attention": list(self.attn),
            "meta": self.meta,
        }
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_queue_put(item_queue: queue.Queue, item, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        try:
            item_queue.put(item, timeout=0.05)
            return True
        except queue.Full:
            try:
                item_queue.get_nowait()
            except queue.Empty:
                pass
    return False


def _ensure_queue_get(item_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            return item_queue.get(timeout=0.05)
        except queue.Empty:
            continue
    return None


def _parse_stream_spec(raw: str) -> CameraStreamSpec:
    """
    Expects `<camera_id>:<kind>:<value>`, e.g. `0:camera:0` or `1:video:/tmp/cam1.mp4`.
    """
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid stream spec '{raw}'. Format: <camera_id>:<kind>:<value>")
    camera_id = int(parts[0])
    kind = parts[1].strip().lower()
    value = parts[2].strip()
    if kind not in {"camera", "video"}:
        raise ValueError(f"Unsupported stream kind '{kind}' (allowed: camera, video)")
    return CameraStreamSpec(camera_id=camera_id, kind=kind, source=value)


def _camera_order(specs: Sequence[CameraStreamSpec]) -> List[int]:
    ids = sorted({spec.camera_id for spec in specs})
    return ids


def _build_capture_reader(spec: CameraStreamSpec):
    if spec.kind == "camera":
        device_id = int(spec.source)
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {device_id}")
        return cap
    if spec.kind == "video":
        path = Path(spec.source).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")
        return cap
    raise ValueError(f"Unsupported stream kind '{spec.kind}'")


def _release_capture(cap: cv2.VideoCapture) -> None:
    cap.release()


# ---------------------------------------------------------------------------
# Worker implementations
# ---------------------------------------------------------------------------


class CameraCaptureThread(threading.Thread):
    """Continuously grabs frames from one source and pushes them into frame queue."""

    def __init__(
        self,
        spec: CameraStreamSpec,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name=f"capture-{spec.camera_id}")
        self._spec = spec
        self._queue = frame_queue
        self._stop = stop_event
        self._frame_index = 0

    def run(self) -> None:
        try:
            cap = _build_capture_reader(self._spec)
        except Exception:
            LOGGER.exception("Capture init failed for camera %s", self._spec.camera_id)
            return

        LOGGER.info("Started capture thread for camera %s (%s)", self._spec.camera_id, self._spec.kind)
        try:
            while not self._stop.is_set():
                success, frame = cap.read()
                if not success or frame is None:
                    if self._spec.kind == "video":
                        LOGGER.info("Video stream for camera %s finished", self._spec.camera_id)
                        break
                    time.sleep(0.01)
                    continue
                timestamp = time.time()
                height, width = frame.shape[:2]
                sample = LiveFrameSample(
                    camera_id=self._spec.camera_id,
                    timestamp=timestamp,
                    frame_index=self._frame_index,
                    csv_row={"source": self._spec.kind},
                    video_path=None,
                    image=np.ascontiguousarray(frame),
                    frame_size=(width, height),
                )
                if not _ensure_queue_put(self._queue, sample, self._stop):
                    break
                self._frame_index += 1
        finally:
            _release_capture(cap)
        LOGGER.info("Capture thread for camera %s stopped", self._spec.camera_id)


class SyncThread(threading.Thread):
    """Builds synchronized batches using the TemporalSynchronizer."""

    def __init__(
        self,
        frame_queue: queue.Queue,
        output_queue: queue.Queue,
        camera_ids: Sequence[int],
        tolerance_sec: float,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="sync-thread")
        self._frames = frame_queue
        self._output = output_queue
        self._stop = stop_event
        self._synchronizer = TemporalSynchronizer(camera_ids, tolerance_sec=tolerance_sec)

    def run(self) -> None:
        LOGGER.info("Synchronization thread started")
        while not self._stop.is_set():
            sample = _ensure_queue_get(self._frames, self._stop)
            if sample is None:
                break
            batch = self._synchronizer.submit(sample)
            if batch is None:
                continue
            _ensure_queue_put(self._output, batch, self._stop)
        LOGGER.info("Synchronization thread stopped")


class EyeCropWorker(mp.Process):
    """Runs EyeCropPipeline on synchronized batches (CPU) in a separate process."""

    def __init__(
        self,
        worker_id: int,
        pipeline_config_path: Path,
        synced_queue: mp.Queue,
        output_queue: mp.Queue,
        camera_order: Sequence[int],
        patch_size: Tuple[int, int],
        stop_event,
    ) -> None:
        super().__init__(daemon=True, name=f"eye-crop-{worker_id}")
        self._worker_id = worker_id
        self._config_path = pipeline_config_path
        self._synced = synced_queue
        self._output = output_queue
        self._camera_order = list(camera_order)
        self._patch_size = patch_size
        self._stop = stop_event
        self._pipeline_config = load_pipeline_config_from_yaml(self._config_path)

    def run(self) -> None:
        LOGGER.info("Eye-crop worker %s starting (CPU, pid=%s)", self._worker_id, self.pid)
        try:
            with EyeCropPipeline(self._pipeline_config) as pipeline:
                while not self._stop.is_set():
                    batch = _ensure_queue_get(self._synced, self._stop)
                    if batch is None:
                        break
                    tri_input = self._process_batch(pipeline, batch)
                    if tri_input is None:
                        continue
                    _ensure_queue_put(self._output, tri_input, self._stop)
        except Exception:
            LOGGER.exception("Eye-crop worker %s crashed", self._worker_id)
        LOGGER.info("Eye-crop worker %s stopped", self._worker_id)

    def _process_batch(self, pipeline: EyeCropPipeline, batch: SyncedBatch) -> Optional[TriCamSample]:
        images: Dict[int, ImageArray] = {}
        frame_sizes: Dict[int, Tuple[int, int]] = {}
        for cam_id, sample in batch.samples.items():
            if not isinstance(sample, LiveFrameSample):
                LOGGER.warning("Unexpected sample type for camera %s; skipping batch", cam_id)
                return None
            images[cam_id] = sample.image
            frame_sizes[cam_id] = sample.frame_size

        pipeline_results: Dict[int, PipelineFrameResult] = {}
        for cam_id, image in images.items():
            sample = batch.samples[cam_id]
            result = pipeline.process(image=image, frame_id=sample.frame_index, timestamp=sample.timestamp)
            pipeline_results[cam_id] = result

        tri_sample = self._build_tricam_sample(batch, pipeline_results, frame_sizes)
        return tri_sample

    def _build_tricam_sample(
        self,
        batch: SyncedBatch,
        pipeline_results: Dict[int, PipelineFrameResult],
        frame_sizes: Dict[int, Tuple[int, int]],
    ) -> Optional[TriCamSample]:
        n_cams = len(self._camera_order)
        n_eyes = n_cams * 2
        patch_h, patch_w = self._patch_size

        eye_patches = np.zeros((1, n_eyes, 1, patch_h, patch_w), dtype=np.float32)
        eye_coords = np.zeros((1, n_eyes, 3), dtype=np.float32)
        eye_coords[:, :, 2] = 1.0  # mark as missing by default

        valid_eye = False
        for cam_idx, cam_id in enumerate(self._camera_order):
            batch_sample = batch.samples.get(cam_id)
            result = pipeline_results.get(cam_id)
            if batch_sample is None or result is None:
                continue
            frame_w, frame_h = frame_sizes[cam_id]
            face_state = self._select_face_state(result)
            if face_state is None or face_state.eye_details is None:
                continue
            details = face_state.eye_details
            left_index = cam_idx * 2
            right_index = left_index + 1
            if self._assign_eye_patch(eye_patches, eye_coords, left_index, details.left_eye_image, details.left_eye_center_frame, frame_w, frame_h):
                valid_eye = True
            if self._assign_eye_patch(eye_patches, eye_coords, right_index, details.right_eye_image, details.right_eye_center_frame, frame_w, frame_h):
                valid_eye = True

        if not valid_eye:
            return None

        meta = {
            "frame_indices": {cam_id: batch.samples[cam_id].frame_index for cam_id in self._camera_order if cam_id in batch.samples},
            "camera_ids": list(self._camera_order),
            "target_timestamp": batch.target_timestamp,
        }
        return TriCamSample(
            timestamp=batch.target_timestamp,
            eye_patches=eye_patches,
            eye_coords=eye_coords,
            meta=meta,
        )

    def _assign_eye_patch(
        self,
        patches: np.ndarray,
        coords: np.ndarray,
        index: int,
        eye_image: Optional[ImageArray],
        center: Optional[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        if eye_image is None or eye_image.size == 0:
            return False
        patch_h, patch_w = self._patch_size
        gray = eye_image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        patches[0, index, 0] = resized.astype(np.float32) / 255.0
        if center is not None and frame_w > 0 and frame_h > 0:
            coords[0, index, 0] = float(center[0]) / float(frame_w)
            coords[0, index, 1] = float(center[1]) / float(frame_h)
            coords[0, index, 2] = 0.0
        else:
            coords[0, index, 2] = 1.0
        return True

    @staticmethod
    def _select_face_state(result: PipelineFrameResult):
        for face in result.faces:
            if face.eye_details and face.eye_details.left_eye_image is not None:
                return face
        for face in result.faces:
            if face.eye_details:
                return face
        return None


class TriCamInferenceWorker(mp.Process):
    """Consumes TriCam-ready samples and runs the neural network (GPU preferred) in its own process."""

    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        stop_event,
        tri_cam_config: TriCamConfig,
        weights_path: Path,
        device: str,
        ema_alpha: Optional[float] = None,
    ) -> None:
        super().__init__(daemon=True, name="tricam-worker")
        self._input = input_queue
        self._output = output_queue
        self._stop = stop_event
        self._config = tri_cam_config
        self._weights = Path(weights_path)
        device_normalized = device.lower()
        use_cuda = torch.cuda.is_available() and device_normalized.startswith("cuda")
        target_device = device if use_cuda or device_normalized == "cpu" else "cpu"
        self._device = torch.device(target_device)
        self._model: Optional[TriCamNet] = None
        self._ema_alpha = ema_alpha
        self._ema: Optional[EMASmoother] = None

    def _initialize_model(self) -> None:
        if self._model is not None:
            return
        self._model = TriCamNet(self._config).to(self._device)
        self._load_weights()
        self._model.eval()
        if self._ema_alpha is not None:
            self._ema = EMASmoother(alpha=self._ema_alpha)

    def _load_weights(self) -> None:
        if not self._weights:
            raise ValueError("TriCam weights path must be provided for inference")
        weights_path = Path(self._weights).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"TriCam weights not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location=self._device)
        state_dict = ckpt.get("model") if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self._model.load_state_dict(state_dict, strict=False)
        LOGGER.info("Loaded TriCam weights from %s", weights_path)

    def run(self) -> None:
        LOGGER.info("TriCam inference worker started on %s (pid=%s)", self._device, self.pid)
        self._initialize_model()
        while not self._stop.is_set():
            sample: TriCamSample = _ensure_queue_get(self._input, self._stop)
            if sample is None:
                break
            try:
                output = self._run_model(sample)
            except Exception:
                LOGGER.exception("TriCam inference failed; dropping sample")
                continue
            if output:
                _ensure_queue_put(self._output, output, self._stop)
        LOGGER.info("TriCam inference worker stopped")

    def _run_model(self, sample: TriCamSample) -> Optional[TriCamOutput]:
        if self._model is None:
            raise RuntimeError("TriCam model not initialized")
        eye_patches = torch.from_numpy(sample.eye_patches).to(self._device)
        eye_coords = torch.from_numpy(sample.eye_coords).to(self._device)
        with torch.no_grad():
            preds = self._model(eye_patches, eye_coords)
            gaze = preds["gaze"]
            attn = preds["attn"]
            if self._ema is not None:
                gaze = self._ema(gaze)
        gaze_xy = gaze.detach().cpu().numpy()[0]
        attn_arr = attn.detach().cpu().numpy()[0]
        return TriCamOutput(
            timestamp=sample.timestamp,
            gaze=(float(gaze_xy[0]), float(gaze_xy[1])),
            attn=attn_arr.tolist(),
            meta=sample.meta,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class OnlineInferenceRunner:
    """High-level orchestrator that wires capture, sync, eye-crop, and TriCam inference."""

    def __init__(
        self,
        streams: Sequence[CameraStreamSpec],
        pipeline_config: Path,
        tricam_weights: Path,
        *,
        sync_tolerance: float = 1.0 / 60.0,
        eye_workers: int = 2,
        patch_size: Tuple[int, int] = (20, 40),
        tricam_device: str = "cuda",
        ema_alpha: Optional[float] = 0.8,
        queue_size: int = 32,
    ) -> None:
        if not streams:
            raise ValueError("At least one stream must be configured")
        self._streams = list(streams)
        self._camera_ids = _camera_order(streams)
        self._pipeline_config = pipeline_config
        self._tricam_weights = tricam_weights
        self._sync_tolerance = sync_tolerance
        self._eye_workers = max(1, eye_workers)
        self._patch_size = patch_size
        self._device = tricam_device
        self._ema_alpha = ema_alpha
        self._queue_size = queue_size

        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        half_size = max(1, queue_size // 2)
        self._sync_queue: mp.Queue = self._ctx.Queue(maxsize=half_size)
        self._crop_queue: mp.Queue = self._ctx.Queue(maxsize=half_size)
        self._result_queue: mp.Queue = self._ctx.Queue(maxsize=queue_size)

        n_cams = len(self._camera_ids)
        self._tricam_config = TriCamConfig(
            n_cams=n_cams,
            patch_h=patch_size[0],
            patch_w=patch_size[1],
            in_ch=1,
        )

        self._threads: List[threading.Thread] = []
        self._processes: List[mp.Process] = []

    def start(self) -> None:
        if self._threads or self._processes:
            return
        self._stop_event.clear()
        # Capture threads
        for spec in self._streams:
            t = CameraCaptureThread(spec, self._frame_queue, self._stop_event)
            t.start()
            self._threads.append(t)

        # Sync thread
        sync_thread = SyncThread(
            self._frame_queue,
            self._sync_queue,
            self._camera_ids,
            self._sync_tolerance,
            self._stop_event,
        )
        sync_thread.start()
        self._threads.append(sync_thread)

        # Eye-crop workers
        for idx in range(self._eye_workers):
            worker = EyeCropWorker(
                worker_id=idx,
                pipeline_config_path=self._pipeline_config,
                synced_queue=self._sync_queue,
                output_queue=self._crop_queue,
                camera_order=self._camera_ids,
                patch_size=self._patch_size,
                stop_event=self._stop_event,
            )
            worker.start()
            self._processes.append(worker)

        # TriCam worker
        tricam_worker = TriCamInferenceWorker(
            self._crop_queue,
            self._result_queue,
            self._stop_event,
            self._tricam_config,
            self._tricam_weights,
            device=self._device,
            ema_alpha=self._ema_alpha,
        )
        tricam_worker.start()
        self._processes.append(tricam_worker)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()
        for proc in self._processes:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
        self._processes.clear()

    def run(self, handler: Callable[[TriCamOutput], None]) -> None:
        self.start()
        LOGGER.info("Online inference runner is live (Ctrl+C to stop)")
        try:
            while not self._stop_event.is_set():
                result = _ensure_queue_get(self._result_queue, self._stop_event)
                if result is None:
                    break
                handler(result)
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run online eye-crop (CPU) + TriCam inference.")
    parser.add_argument(
        "--stream",
        action="append",
        required=True,
        help="Stream spec in the form <camera_id>:<kind>:<value> (e.g. 0:camera:0, 1:video:/tmp/cam1.mp4)",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("data_process/araya_eye/app/config/config_simple.yaml"),
        help="EyeCropPipeline config YAML.",
    )
    parser.add_argument(
        "--tricam-weights",
        type=Path,
        required=True,
        help="Path to TriCam checkpoint (.pt).",
    )
    parser.add_argument(
        "--sync-tolerance",
        type=float,
        default=1.0 / 60.0,
        help="Maximum timestamp delta (seconds) allowed between cameras.",
    )
    parser.add_argument(
        "--eye-workers",
        type=int,
        default=2,
        help="Number of eye-crop worker threads (CPU).",
    )
    parser.add_argument(
        "--patch-size",
        type=str,
        default="20x40",
        help="Eye patch size HxW for TriCamNet (e.g. 20x40).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="TriCam inference device (cuda or cpu).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.8,
        help="EMA smoothing factor for gaze output (set negative to disable).",
    )
    return parser


def _parse_patch_size(raw: str) -> Tuple[int, int]:
    try:
        h_str, w_str = raw.lower().split("x")
        return int(h_str), int(w_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid patch size '{raw}'") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    streams = [_parse_stream_spec(spec) for spec in args.stream]
    patch_size = _parse_patch_size(args.patch_size)
    ema = args.ema_alpha if args.ema_alpha > 0 else None

    runner = OnlineInferenceRunner(
        streams=streams,
        pipeline_config=args.pipeline_config,
        tricam_weights=args.tricam_weights,
        sync_tolerance=args.sync_tolerance,
        eye_workers=args.eye_workers,
        patch_size=patch_size,
        tricam_device=args.device,
        ema_alpha=ema,
    )

    def _print_handler(result: TriCamOutput) -> None:
        print(result.to_json(), flush=True)

    def _signal_handler(signum, frame):
        LOGGER.info("Received signal %s, shutting down...", signum)
        runner.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    runner.run(_print_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
