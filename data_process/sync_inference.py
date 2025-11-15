"""Offline synchronization + inference pipeline with hooks for online reuse."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import cv2
import numpy as np
from omegaconf import OmegaConf

from data_process.sync_pipeline import (
    CameraArtifacts,
    FrameSample,
    SyncedBatch,
    discover_camera_artifacts,
    synchronize_session,
)
from data_process.araya_eye.core.dtypes.media import ImageArray
from data_process.araya_eye.core.integrators.blink_detector.api import EARBlinkDetails
from data_process.araya_eye.core.integrators.face_detector.wrappers.yolov9_wb25.config import (
    YoloV9Wb25FaceDetectorConfig,
)
from data_process.araya_eye.core.integrators.face_recognizer.wrappers.iou_tracker.config import (
    IouTrackerConfig,
)
from data_process.araya_eye.core.models.mp_facemesh import MpFacemeshConfig
from data_process.araya_eye.core.pipeline import EyeCropPipeline, PipelineFrameResult, PipelineFaceState
from data_process.araya_eye.core.pipeline.config import PipelineConfig
from data_process.araya_eye.core.integrators.face_detector.config import FaceDetectorConfig
from data_process.araya_eye.core.integrators.face_recognizer.config import FaceRecognizerConfig


@dataclass
class BatchInferenceResult:
    batch: SyncedBatch
    camera_results: Dict[int, PipelineFrameResult]

    @property
    def target_timestamp(self) -> float:
        return self.batch.target_timestamp

    @property
    def max_delta(self) -> float:
        return self.batch.max_delta


@dataclass
class EyeCropRecord:
    session: str
    camera_id: int
    frame_index: int
    timestamp: float
    face_index: int
    person_id: Optional[int]
    eye_side: Literal["left", "right"]
    csv_row: Dict[str, str]
    image_path: str
    image_npy_path: Optional[str] = None
    ear: Optional[float] = None
    bbox: Optional[Dict[str, object]] = None
    eye_center_x: Optional[float] = None
    eye_center_y: Optional[float] = None
    camera_position: Optional[str] = None
    detected: bool = True


@dataclass
class FaceCropRecord:
    session: str
    camera_id: int
    frame_index: int
    timestamp: float
    face_index: int
    csv_row: Dict[str, str]
    bbox: Dict[str, object]
    image_path: str
    image_npy_path: Optional[str] = None
    camera_position: Optional[str] = None


class FrameProvider:
    """Abstract frame provider so we can reuse the inference stage online."""

    def get_frame(self, sample: FrameSample) -> ImageArray:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface hook
        pass


class FrameOutOfRangeError(RuntimeError):
    def __init__(self, camera_id: int, requested_index: int, max_index: int, video_path: str):
        super().__init__(
            f"Camera {camera_id}: requested frame {requested_index} but video only has up to {max_index} "
            f"({video_path})"
        )
        self.camera_id = camera_id
        self.requested_index = requested_index
        self.max_index = max_index
        self.video_path = video_path


class VideoFrameProvider(FrameProvider):
    """Reads frames on-demand from recorded video files."""

    def __init__(self, camera_videos: Dict[int, Path]):
        self._videos = {cam_id: Path(path).expanduser().resolve() for cam_id, path in camera_videos.items()}
        self._captures: Dict[int, cv2.VideoCapture] = {}
        self._last_index: Dict[int, int] = {}
        self._frame_counts: Dict[int, int] = {}
        self._warned_bounds: Dict[int, bool] = {}

    def __enter__(self) -> "VideoFrameProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_frame(self, sample: FrameSample) -> ImageArray:
        path = self._videos.get(sample.camera_id)
        if path is None:
            raise KeyError(f"No video registered for camera {sample.camera_id}")
        cap = self._captures.get(sample.camera_id)
        if cap is None:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {path}")
            self._captures[sample.camera_id] = cap
            self._last_index[sample.camera_id] = -1
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._frame_counts[sample.camera_id] = frame_count

        desired_index = sample.frame_index
        last_index = self._last_index.get(sample.camera_id, -1)
        frame_count = self._frame_counts.get(sample.camera_id, 0)
        if frame_count > 0 and desired_index >= frame_count:
            raise FrameOutOfRangeError(
                camera_id=sample.camera_id,
                requested_index=desired_index,
                max_index=frame_count - 1,
                video_path=str(path),
            )

        if desired_index != last_index + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, desired_index)
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError(f"Unable to read frame {desired_index} from {path}")
        self._last_index[sample.camera_id] = desired_index
        return frame

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        self._last_index.clear()
        self._frame_counts.clear()
        self._warned_bounds.clear()


class SyncInferenceEngine:
    """Runs the araya_eye pipeline on synchronized batches."""

    def __init__(self, pipeline_config: PipelineConfig):
        self._pipeline = EyeCropPipeline(pipeline_config)

    def __enter__(self) -> "SyncInferenceEngine":
        self._pipeline.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._pipeline.__exit__(exc_type, exc_val, exc_tb)

    def process_batch(
        self,
        batch: SyncedBatch,
        images: Dict[int, ImageArray],
    ) -> BatchInferenceResult:
        camera_results: Dict[int, PipelineFrameResult] = {}
        for cam_id, image in images.items():
            sample = batch.samples[cam_id]
            result = self._pipeline.process(
                image=image,
                frame_id=sample.frame_index,
                timestamp=sample.timestamp,
            )
            camera_results[cam_id] = result
        return BatchInferenceResult(batch=batch, camera_results=camera_results)


class CameraPositionEstimator:
    """Estimate relative camera placement (left/center/right) using face centers."""

    def __init__(self, camera_ids: Iterable[int]):
        ids = list(camera_ids)
        if not ids:
            raise ValueError("CameraPositionEstimator requires at least one camera id.")
        self.camera_ids = ids
        self._stats: Dict[int, Dict[str, float]] = {
            cam_id: {"sum": 0.0, "count": 0.0} for cam_id in ids
        }

    def update(self, camera_id: int, faces: Sequence[PipelineFaceState], image_width: int) -> None:
        if image_width <= 0 or camera_id not in self._stats:
            return
        centers: List[float] = []
        for face in faces:
            bbox = face.bbox
            center = (bbox.x + bbox.width / 2.0) / float(image_width)
            centers.append(center)
        if not centers:
            return
        stats = self._stats[camera_id]
        stats["sum"] += sum(centers) / len(centers)
        stats["count"] += 1

    def update_from_results(
        self,
        camera_results: Dict[int, PipelineFrameResult],
        images: Dict[int, ImageArray],
    ) -> None:
        for cam_id, result in camera_results.items():
            image = images.get(cam_id)
            if image is None:
                continue
            width = image.shape[1] if len(image.shape) >= 2 else 0
            self.update(cam_id, result.faces, width)

    def camera_labels(self) -> Dict[int, str]:
        averages: List[Tuple[int, float]] = []
        for cam_id, stats in self._stats.items():
            if stats["count"] <= 0:
                continue
            averages.append((cam_id, stats["sum"] / stats["count"]))
        if len(averages) < len(self.camera_ids):
            return {}
        ordered = sorted(averages, key=lambda item: item[1])
        label_names = self._label_names(len(ordered))
        return {cam_id: label_names[idx] for idx, (cam_id, _) in enumerate(ordered)}

    @staticmethod
    def _label_names(count: int) -> List[str]:
        if count == 1:
            return ["center"]
        if count == 2:
            return ["left", "right"]
        if count == 3:
            return ["left", "center", "right"]
        return [f"position_{i}" for i in range(count)]


def _resolve_path(base: Path, path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    candidates = [
        base / path,
        base.parent / path,
        base.parent.parent / path,
    ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return str(resolved)
    return str((base / path).expanduser().resolve())


def _load_face_detector_config(config_dict: Dict[str, object], base_dir: Path) -> FaceDetectorConfig:
    detector_type = config_dict.get("type")
    if detector_type != "yolov9_wb25":
        raise NotImplementedError(f"Unsupported face detector type: {detector_type}")
    yolo_cfg = dict(config_dict["yolov9_wb25"])
    yolo_cfg["model_path"] = _resolve_path(base_dir, yolo_cfg["model_path"])
    if "input_size" in yolo_cfg:
        yolo_cfg["input_size"] = tuple(yolo_cfg["input_size"])
    return FaceDetectorConfig(
        type="yolov9_wb25",
        yolov9_wb25=YoloV9Wb25FaceDetectorConfig(**yolo_cfg),
    )


def _load_face_recognizer_config(config_dict: Dict[str, object]) -> FaceRecognizerConfig:
    recognizer_type = config_dict.get("type")
    if recognizer_type != "iou_tracker":
        raise NotImplementedError(f"Unsupported face recognizer type: {recognizer_type}")
    tracker_cfg = dict(config_dict["iou_tracker"])
    keys = [
        "iou_threshold",
        "max_lost_frames",
        "min_track_length",
        "max_persons",
        "confidence_threshold",
        "merge_threshold",
    ]
    filtered = {k: tracker_cfg[k] for k in keys if k in tracker_cfg}
    missing = [k for k in keys if k not in filtered]
    if missing:
        raise ValueError(f"Missing IOU tracker keys: {missing}")
    optional_defaults = {
        "track_length_weight": 0.5,
        "distance_penalty_factor": 1.0,
    }
    for name, default in optional_defaults.items():
        filtered[name] = tracker_cfg.get(name, default)
    return FaceRecognizerConfig(
        type="iou_tracker",
        iou_tracker=IouTrackerConfig(**filtered),
    )


def load_pipeline_config_from_yaml(config_path: Path) -> PipelineConfig:
    # Ensure Hydra-style resolvers are available
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda pattern="": datetime.now().strftime(pattern) if pattern else datetime.now().isoformat(),
        )
    config_path = config_path.expanduser().resolve()
    if not OmegaConf.has_resolver("hydra"):
        def _hydra_resolver(attribute: str = "") -> str:
            base = config_path.parent
            mapping = {
                "runtime.cwd": str(base),
                "job.name": config_path.stem,
                "run.dir": str(base / "outputs" / config_path.stem),
                "sweep.dir": str(base / "multirun" / config_path.stem),
                "sweep.subdir": "0",
            }
            return mapping.get(attribute, str(base))
        OmegaConf.register_new_resolver("hydra", _hydra_resolver)

    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Invalid configuration format.")
    pipeline_section = cfg_dict.get("pipeline", cfg_dict)
    detector_cfg = _load_face_detector_config(pipeline_section["face_detector"], config_path.parent)
    recognizer_cfg = _load_face_recognizer_config(pipeline_section["face_recognizer"])
    mp_cfg_dict = pipeline_section.get("mp_facemesh")
    mp_cfg_dict = pipeline_section.get("mp_facemesh")
    if mp_cfg_dict is None:
        blink_cfg = pipeline_section.get("blink_detector", {}).get("lstm_base_01_blink_detector", {})
        mp_cfg_dict = blink_cfg.get("mp_facemesh")
    if mp_cfg_dict is None:
        raise ValueError("Could not find mp_facemesh configuration in the provided config file.")
    mp_cfg = MpFacemeshConfig(**mp_cfg_dict)

    return PipelineConfig(
        face_detector=asdict(detector_cfg),
        face_recognizer=asdict(recognizer_cfg),
        mp_facemesh=asdict(mp_cfg),
    )


def _serialize_face_state(state: PipelineFaceState) -> Dict[str, object]:
    data: Dict[str, object] = {
        "bbox": asdict(state.bbox),
        "recognition": None,
        "eye_details": None,
    }
    if state.recognition:
        data["recognition"] = {
            "person_id": state.recognition.person_id,
            "confidence": state.recognition.confidence,
        }
    if state.eye_details:
        data["eye_details"] = _serialize_ear_details(state.eye_details)
    return data


def _serialize_ear_details(details: EARBlinkDetails) -> Dict[str, object]:
    return {
        "person_id": details.person_id,
        "left_eye": {
            "bbox": _serialize_eye_bb(details.left_eye_bb),
            "center_frame": details.left_eye_center_frame,
        },
        "right_eye": {
            "bbox": _serialize_eye_bb(details.right_eye_bb),
            "center_frame": details.right_eye_center_frame,
        },
    }


def _serialize_eye_bb(bb) -> Optional[Dict[str, object]]:
    if bb is None:
        return None
    payload = asdict(bb)
    try:
        payload["ear"] = bb.ear
    except Exception:
        payload["ear"] = None
    return payload


def _serialize_pipeline_frame(result: PipelineFrameResult) -> Dict[str, object]:
    return {
        "processing_time": {
            "face_detector": result.face_detector.processing_time,
            "face_recognizer": result.face_recognizer.processing_time,
            "mediapipe": result.mediapipe.processing_time,
            "total": result.total_processing_time,
        },
        "faces": [_serialize_face_state(state) for state in result.faces],
    }


def serialize_batch_result(result: BatchInferenceResult) -> Dict[str, object]:
    record: Dict[str, object] = {
        "target_timestamp": result.target_timestamp,
        "max_delta": result.max_delta,
        "cameras": {},
    }
    for cam_id, sample in result.batch.samples.items():
        camera_entry = {
            "frame_index": sample.frame_index,
            "timestamp": sample.timestamp,
            "csv": sample.csv_row,
        }
        if sample.video_path:
            camera_entry["video_path"] = str(sample.video_path)
        if cam_id in result.camera_results:
            camera_entry["pipeline"] = _serialize_pipeline_frame(result.camera_results[cam_id])
        record["cameras"][str(cam_id)] = camera_entry
    return record


def _save_eye_image(
    image_root: Path,
    camera_id: int,
    frame_index: int,
    face_index: int,
    eye_side: Literal["left", "right"],
    image: ImageArray,
) -> Tuple[Path, Path, Path]:
    camera_dir = image_root / f"camera{camera_id:02d}"
    frame_dir = camera_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    base_path = frame_dir / f"face{face_index:02d}_{eye_side}"

    png_path = base_path.with_suffix(".png")
    if not cv2.imwrite(str(png_path), image):
        raise RuntimeError(f"Failed to write eye crop: {png_path}")

    npy_path = base_path.with_suffix(".npy")
    np.save(str(npy_path), image, allow_pickle=False)

    return png_path, npy_path


def _save_face_image(
    image_root: Path,
    camera_id: int,
    frame_index: int,
    face_index: int,
    image: ImageArray,
) -> Tuple[Path, Path]:
    camera_dir = image_root / f"camera{camera_id:02d}"
    frame_dir = camera_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    base_path = frame_dir / f"face{face_index:02d}"
    png_path = base_path.with_suffix(".png")
    if not cv2.imwrite(str(png_path), image):
        raise RuntimeError(f"Failed to write face crop: {png_path}")
    npy_path = base_path.with_suffix(".npy")
    np.save(str(npy_path), image, allow_pickle=False)
    return png_path, npy_path


def _collect_eye_crop_records(
    session_name: str,
    batch: SyncedBatch,
    camera_results: Dict[int, PipelineFrameResult],
    image_root: Path,
    camera_labels: Dict[int, str],
) -> Tuple[List[EyeCropRecord], int]:
    records: List[EyeCropRecord] = []
    failed = 0
    output_root = image_root.parent
    for cam_id, result in camera_results.items():
        sample = batch.samples[cam_id]
        for face_idx, face_state in enumerate(result.faces):
            details = face_state.eye_details
            if not details:
                failed += 2
                continue
            for eye_side, eye_image, eye_bb in (
                ("left", details.left_eye_image, details.left_eye_bb),
                ("right", details.right_eye_image, details.right_eye_bb),
            ):
                if eye_image is None:
                    failed += 1
                    continue
                png_path, npy_path = _save_eye_image(
                    image_root, cam_id, sample.frame_index, face_idx, eye_side, eye_image
                )
                rel_png = _make_relative_path(png_path, output_root)
                rel_npy = _make_relative_path(npy_path, output_root)
                bbox_payload = _serialize_eye_bb(eye_bb)
                ear_value = bbox_payload.get("ear") if bbox_payload else None
                center = (
                    details.left_eye_center_frame
                    if eye_side == "left"
                    else details.right_eye_center_frame
                )
                detected = center is not None and center[0] >= 0.0 and center[1] >= 0.0
                if not detected:
                    failed += 1
                records.append(
                    EyeCropRecord(
                        session=session_name,
                        camera_id=cam_id,
                        frame_index=sample.frame_index,
                        timestamp=sample.timestamp,
                        face_index=face_idx,
                        person_id=details.person_id,
                        eye_side=eye_side,  # type: ignore[arg-type]
                        image_path=str(rel_png),
                        image_npy_path=str(rel_npy),
                        ear=ear_value,
                        bbox=bbox_payload,
                        csv_row=sample.csv_row,
                        eye_center_x=center[0] if center else None,
                        eye_center_y=center[1] if center else None,
                        camera_position=camera_labels.get(cam_id),
                        detected=detected,
                    )
                )
    return records, failed


def _collect_face_crop_records(
    session_name: str,
    batch: SyncedBatch,
    camera_results: Dict[int, PipelineFrameResult],
    images: Dict[int, ImageArray],
    image_root: Path,
    camera_labels: Dict[int, str],
) -> List[FaceCropRecord]:
    records: List[FaceCropRecord] = []
    output_root = image_root.parent
    for cam_id, result in camera_results.items():
        sample = batch.samples[cam_id]
        frame = images.get(cam_id)
        if frame is None:
            continue
        for face_idx, face_state in enumerate(result.faces):
            bbox = face_state.bbox
            x1 = max(0, int(bbox.x))
            y1 = max(0, int(bbox.y))
            x2 = min(frame.shape[1], int(bbox.x + bbox.width))
            y2 = min(frame.shape[0], int(bbox.y + bbox.height))
            if x2 <= x1 or y2 <= y1:
                continue
            face_patch = frame[y1:y2, x1:x2]
            if face_patch.size == 0:
                continue
            png_path, npy_path = _save_face_image(image_root, cam_id, sample.frame_index, face_idx, face_patch)
            rel_png = _make_relative_path(png_path, output_root)
            rel_npy = _make_relative_path(npy_path, output_root)
            records.append(
                FaceCropRecord(
                    session=session_name,
                    camera_id=cam_id,
                    frame_index=sample.frame_index,
                    timestamp=sample.timestamp,
                    face_index=face_idx,
                    csv_row=sample.csv_row,
                    bbox=asdict(bbox),
                    image_path=str(rel_png),
                    image_npy_path=str(rel_npy),
                    camera_position=camera_labels.get(cam_id),
                )
            )
    return records


def _make_relative_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _prepare_frame_provider(artifacts: Iterable[CameraArtifacts]) -> VideoFrameProvider:
    videos: Dict[int, Path] = {}
    for artifact in artifacts:
        if not artifact.video_path:
            raise RuntimeError(f"No video file found for camera {artifact.camera_id}")
        videos[artifact.camera_id] = artifact.video_path
    return VideoFrameProvider(videos)


def run_sync_and_infer(
    session_dir: Path,
    config_path: Path,
    *,
    tolerance_sec: float = 1.0 / 30.0,
    output_path: Optional[Path] = None,
    limit_batches: Optional[int] = None,
) -> Tuple[Path, int]:
    session_dir = session_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    pipeline_config = load_pipeline_config_from_yaml(config_path)
    artifacts = discover_camera_artifacts(session_dir)
    provider = _prepare_frame_provider(artifacts)
    batches = synchronize_session(session_dir, tolerance_sec=tolerance_sec)

    output_path = (output_path or (session_dir / "synced_results.jsonl")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with provider, SyncInferenceEngine(pipeline_config) as engine, output_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            try:
                images = {cam_id: provider.get_frame(sample) for cam_id, sample in batch.samples.items()}
            except FrameOutOfRangeError as exc:
                print(f"Skipping batch at {batch.target_timestamp:.3f}s due to missing video frames: {exc}")
                continue
            result = engine.process_batch(batch, images)
            handle.write(json.dumps(serialize_batch_result(result), ensure_ascii=False) + "\n")
            processed += 1
            if limit_batches and processed >= limit_batches:
                break

    return output_path, processed


def run_sync_and_extract_eyes(
    session_dir: Path,
    config_path: Path,
    *,
    tolerance_sec: float = 1.0 / 30.0,
    output_dir: Optional[Path] = None,
    limit_batches: Optional[int] = None,
) -> Tuple[Path, int, int, int, int]:
    """Synchronize, run MediaPipe, and persist cropped eye images plus metadata."""
    session_dir = session_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    pipeline_config = load_pipeline_config_from_yaml(config_path)
    artifacts = discover_camera_artifacts(session_dir)
    provider = _prepare_frame_provider(artifacts)
    batches = synchronize_session(session_dir, tolerance_sec=tolerance_sec)

    base_output_dir = (output_dir or session_dir).expanduser().resolve()
    image_root = base_output_dir / "eye_images"
    face_image_root = base_output_dir / "face_images"
    manifest_path = base_output_dir / "eye_crops.jsonl"
    face_manifest_path = base_output_dir / "face_crops.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    session_name = session_dir.name

    processed_batches = 0
    saved_images = 0
    saved_faces = 0
    failed_eyes = 0
    estimator = CameraPositionEstimator([art.camera_id for art in artifacts])
    with (
        provider,
        SyncInferenceEngine(pipeline_config) as engine,
        manifest_path.open("w", encoding="utf-8") as eye_handle,
        face_manifest_path.open("w", encoding="utf-8") as face_handle,
    ):
        for batch in batches:
            try:
                images = {cam_id: provider.get_frame(sample) for cam_id, sample in batch.samples.items()}
            except FrameOutOfRangeError as exc:
                print(f"Skipping batch at {batch.target_timestamp:.3f}s due to missing video frames: {exc}")
                continue
            inference = engine.process_batch(batch, images)
            estimator.update_from_results(inference.camera_results, images)
            camera_labels = estimator.camera_labels()
            records, failed = _collect_eye_crop_records(session_name, batch, inference.camera_results, image_root, camera_labels)
            face_records = _collect_face_crop_records(
                session_name, batch, inference.camera_results, images, face_image_root, camera_labels
            )
            for record in records:
                eye_handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            for face_record in face_records:
                face_handle.write(json.dumps(asdict(face_record), ensure_ascii=False) + "\n")
            processed_batches += 1
            saved_images += len(records)
            saved_faces += len(face_records)
            failed_eyes += failed
            if limit_batches and processed_batches >= limit_batches:
                break

    return manifest_path, processed_batches, saved_images, saved_faces, failed_eyes
