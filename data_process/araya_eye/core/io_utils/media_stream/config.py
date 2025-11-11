from dataclasses import dataclass
from typing import Literal, Optional, List
from pathlib import Path
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
@dataclass
class CameraStreamConfig:
    video_id: int


@pydantic_dataclass
@dataclass
class VideoStreamConfig:
    video_path: str


@pydantic_dataclass
@dataclass
class ImageDirectoryConfig:
    image_dir: str
    sort_method: Literal["natural", "name", "time"]
    fps: float
    skip_paths: Optional[List[str]] = None


@pydantic_dataclass
@dataclass
class SocketStreamConfig:
    host: str
    port: int


@pydantic_dataclass
@dataclass
class MediaStreamConfig:
    type: Literal["camera", "video", "directory", "socket"]
    camera: Optional[CameraStreamConfig] = None
    video: Optional[VideoStreamConfig] = None
    directory: Optional[ImageDirectoryConfig] = None
    socket: Optional[SocketStreamConfig] = None