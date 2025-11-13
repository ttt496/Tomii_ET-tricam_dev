from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Literal, Union
from pathlib import Path
import cv2
import socket
import struct
import pickle
import time
import re


class TimestampManager:
    def __init__(self, max_history: int = 60):
        self.timestamps: List[float] = []
        self.max_history = max_history
        self.start_time: Optional[float] = None
    
    def add_timestamp(self) -> None:
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.max_history:
            self.timestamps.pop(0)
    
    def set_start_time(self) -> None:
        self.start_time = time.time()
    
    def get_current_time(self) -> float:
        return 0.0 if self.start_time is None else time.time() - self.start_time
    
    def calculate_fps(self) -> float:
        if len(self.timestamps) < 2: return 0.0
        durations = [t2 - t1 for t1, t2 in zip(self.timestamps, self.timestamps[1:])]
        avg_duration = sum(durations) / len(durations)
        return 1.0 / avg_duration if avg_duration > 0 else 0.0


class MediaStreamReader(ABC):    
    @abstractmethod
    def __enter__(self) -> MediaStreamReader: pass
    
    @abstractmethod
    def __exit__(self, *args) -> Optional[bool]: pass
    
    @abstractmethod
    def __iter__(self): pass
    
    @property
    @abstractmethod
    def fps(self) -> float: pass

    @property
    @abstractmethod
    def current_time(self) -> float: pass

    @property
    @abstractmethod
    def initial_size(self) -> Tuple[int, int]: pass # (width, height)

    @property
    @abstractmethod
    def current_size(self) -> Tuple[int, int]: pass # (width, height)


class CameraStreamReader(MediaStreamReader):
    def __init__(self, video_id: int):
        self.video_id = video_id
        self.cap = None
        self._timestamp_mgr = TimestampManager()
        self._initial_size:Optional[Tuple[int,int]] = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_id)
        if not self.cap.isOpened():
            raise ValueError(f'ERROR: Cannot open camera {self.video_id}')
        return self

    def __exit__(self, *args):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __iter__(self):
        if not self.cap: return
        self._timestamp_mgr.set_start_time()
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            self._timestamp_mgr.add_timestamp()
            if self._initial_size == None: self._initial_size = self.current_size
            yield frame

    @property
    def current_time(self) -> float:
        return self._timestamp_mgr.get_current_time()

    @property
    def fps(self) -> float:
        return self._timestamp_mgr.calculate_fps()

    @property
    def initial_size(self) -> Tuple[int, int]:
        if self._initial_size == None: return (0, 0)
        return self._initial_size

    @property
    def current_size(self) -> Tuple[int, int]:
        if not self.cap: return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)


class VideoStreamReader(MediaStreamReader):
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.current_frame_index: Optional[int] = None
        self.cap = None
        self._initial_size:Optional[Tuple[int,int]] = None

    def __enter__(self):
        if not self.video_path.exists():
            raise ValueError(f'Video file not found: {self.video_path}')
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        return self

    def __exit__(self, *args):
        if self.cap:
            self.cap.release()
            self.cap = None

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0.0

    @property
    def initial_size(self) -> Tuple[int, int]:
        if self._initial_size == None: return (0, 0)
        return self._initial_size

    @property
    def current_size(self) -> Tuple[int, int]:
        if not self.cap: return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    @property
    def current_time(self) -> float:
        if self.current_frame_index is None: return 0.0
        return self.current_frame_index / self.fps if self.fps > 0 else 0.0

    def __iter__(self):
        assert self.cap
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            self.current_frame_index = frame_idx
            if self._initial_size == None: self._initial_size = self.current_size
            yield frame
            frame_idx += 1


class ImageDirectoryReader(MediaStreamReader):
    _IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    SORT_METHOD = Literal["natural","name","time"]
    
    def __init__(self, image_dir:str, sort_method:SORT_METHOD, fps:float, skip_paths:Optional[List[str]]=None):
        # Validate directory existence
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists(): raise ValueError(f'Image directory not found: {image_dir}')
        if not self.image_dir.is_dir(): raise ValueError(f'Path is not a directory: {image_dir}')
        
        self.image_paths = self.generate_image_paths(sort_method, skip_paths)
        if not self.image_paths: raise ValueError(f"No image files found in directory: {image_dir}")

        self._fps = fps
        self.current_frame_index: Optional[int] = None
        self._initial_size: Optional[Tuple[int, int]] = None
        self._current_size: Optional[Tuple[int, int]] = None
        self._timestamp_mgr = TimestampManager()

    def __enter__(self): return self
    def __exit__(self, *args): pass
    
    def generate_image_paths(self, sort_method:SORT_METHOD, skip_paths:Optional[List[str]]) -> List[Path]:
        image_paths = [
            p for p in self.image_dir.iterdir() 
            if p.is_file() and p.suffix.lower() in self._IMAGE_EXTENSIONS
        ]
        
        # Skip paths filtering
        if skip_paths:
            skip_set = {Path(p).resolve() for p in skip_paths}
            image_paths = [p for p in image_paths if p.resolve() not in skip_set]
        
        if sort_method == "natural":
            def natural_sort_key(path: Path) -> List[Union[str,int]]:
                """Natural sorting key for frame numbers (frame1, frame2, frame10)"""
                parts:List[Union[str,int]] = []
                for part in re.split(r'(\d+)', path.stem):
                    if part.isdigit(): parts.append(int(part))
                    else: parts.append(part.lower())
                return parts
            image_paths.sort(key=natural_sort_key)
        elif sort_method == "name":
            image_paths.sort(key=lambda p: p.name.lower())
        elif sort_method == "time":
            image_paths.sort(key=lambda p: p.stat().st_mtime)
        
        return image_paths

    @property
    def fps(self) -> float: return self._fps

    @property
    def initial_size(self) -> Tuple[int, int]:
        if self._initial_size == None: return (0, 0)
        return self._initial_size

    @property
    def current_size(self) -> Tuple[int, int]:
        if self._current_size == None: return (0, 0)
        return self._current_size

    @property
    def current_time(self) -> float:
        if self.current_frame_index is None: return 0.0
        return self.current_frame_index / self._fps

    def __iter__(self):
        self._timestamp_mgr.set_start_time()
        for frame_idx, image_path in enumerate(self.image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"WARNING: Failed to load image: {image_path}")
                continue
            self.current_frame_index = frame_idx
            height, width = frame.shape[:2]
            if self._initial_size == None: self._initial_size = (width, height)
            self._current_size = (width, height)
            
            self._timestamp_mgr.add_timestamp()
            yield frame


class SocketStreamReader(MediaStreamReader):
    """ソケット通信による動画受信"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8485):
        self.host = host
        self.port = port
        self.server_socket = None
        self.conn = None
        self._timestamp_mgr = TimestampManager()

        self._initial_size: Optional[Tuple[int,int]] = None
        self._current_size: Optional[Tuple[int,int]] = None

    def __enter__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"INFO: Listening on {self.host}:{self.port}")
        self.conn, addr = self.server_socket.accept()
        print(f"INFO: Connected by {addr}")
        return self

    def __exit__(self, *args):
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None

    def _recv_all(self, size: int) -> bytes:
        """指定サイズのデータを完全受信"""
        if not self.conn: raise ConnectionError("No connection")
        data = b''
        while len(data) < size:
            packet = self.conn.recv(size - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken")
            data += packet
        return data

    @property
    def current_time(self) -> float:
        return self._timestamp_mgr.get_current_time()

    def __iter__(self):
        payload_size = struct.calcsize("!I")
        self._timestamp_mgr.set_start_time()
        try:
            while True:
                # サイズ情報受信
                packed_size = self._recv_all(payload_size)
                frame_size = struct.unpack("!I", packed_size)[0]

                # フレームデータ受信
                frame_data = self._recv_all(frame_size)
                frame = pickle.loads(frame_data)
                
                # フレームサイズを動的更新
                if frame is not None and hasattr(frame, 'shape') and len(frame.shape) >= 2:
                    height, width = frame.shape[:2]
                    if self._initial_size == None: self._initial_size = (width, height)
                    self._current_size = (width, height)
                
                self._timestamp_mgr.add_timestamp()
                yield frame
        except (ConnectionError, socket.error) as e:
            print(f"WARNING: Socket connection error: {e}")

    @property
    def initial_size(self) -> Tuple[int, int]:
        if self._initial_size == None: return (0, 0)
        return self._initial_size

    @property
    def current_size(self) -> Tuple[int, int]:
        if self._current_size == None: return (0, 0)
        return self._current_size

    @property
    def fps(self) -> float:
        return self._timestamp_mgr.calculate_fps()
