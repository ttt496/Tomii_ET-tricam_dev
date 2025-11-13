"""Utilities for capturing, visualizing, and analysing webcam streams."""

from .calibration import (
    CalibrationEvent,
    CalibrationPhaseState,
    CalibrationPoint,
    CalibrationRecordingResult,
    CalibrationRenderer,
    CalibrationSequencer,
    CameraFrameLog,
    build_calibration_points,
    record_calibration_session,
)
from .capture import list_available_cameras, record_from_cameras
from .eye_tracking import EyeTracker, FaceEyes, EyeRegion, run_eye_tracking
from .obs_tools import (
    CameraIndexInfo,
    CameraMapping,
    OBSInputInfo,
    OBSWebSocketError,
    fetch_obs_camera_inputs,
    map_obs_inputs_to_indices,
    probe_camera_indices,
)
from .align import run_alignment_ui
from .visualize import live_preview, playback_recordings

__all__ = [
    "list_available_cameras",
    "record_from_cameras",
    "live_preview",
    "playback_recordings",
    "EyeTracker",
    "FaceEyes",
    "EyeRegion",
    "run_eye_tracking",
    "record_calibration_session",
    "CalibrationPoint",
    "CalibrationSequencer",
    "CalibrationRenderer",
    "CalibrationPhaseState",
    "CalibrationEvent",
    "CalibrationRecordingResult",
    "CameraFrameLog",
    "build_calibration_points",
    "OBSInputInfo",
    "CameraIndexInfo",
    "CameraMapping",
    "OBSWebSocketError",
    "fetch_obs_camera_inputs",
    "map_obs_inputs_to_indices",
    "probe_camera_indices",
    "run_alignment_ui",
]
