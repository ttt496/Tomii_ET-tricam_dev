from .factory import create_media_stream_reader
from .config import (
    MediaStreamConfig,
    CameraStreamConfig,
    VideoStreamConfig,
    ImageDirectoryConfig,
    SocketStreamConfig
)

# Core abstract class only
from .media_stream import MediaStreamReader

# Public API - factory pattern encouraged
__all__ = [
    "create_media_stream_reader",
    "MediaStreamConfig",
    "CameraStreamConfig",
    "VideoStreamConfig",
    "ImageDirectoryConfig", 
    "SocketStreamConfig",
    "MediaStreamReader"
]