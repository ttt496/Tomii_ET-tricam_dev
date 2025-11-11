from pathlib import Path
from .config import MediaStreamConfig
from .media_stream import MediaStreamReader, CameraStreamReader, VideoStreamReader, ImageDirectoryReader, SocketStreamReader


def create_media_stream_reader(config: MediaStreamConfig) -> MediaStreamReader:
    """Factory function to create MediaStreamReader based on configuration"""
    
    if config.type == "camera":
        assert config.camera
        return CameraStreamReader(config.camera.video_id)
    
    if config.type == "video":
        assert config.video
        return VideoStreamReader(config.video.video_path)
    
    if config.type == "directory":
        assert config.directory
        reader = ImageDirectoryReader(config.directory.image_dir)
        if config.directory.skip_paths:
            reader.add_skip_paths(config.directory.skip_paths)
        return reader
    
    if config.type == "socket":
        assert config.socket
        return SocketStreamReader(config.socket.host, config.socket.port)

    raise ValueError(f"Unknown stream type: {config.type}")