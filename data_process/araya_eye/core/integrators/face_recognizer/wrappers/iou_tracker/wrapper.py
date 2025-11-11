import time
from typing import List, Dict
from dataclasses import dataclass, field

from core.integrators.face_detector import FaceBoundingBox
from core.dtypes.person import PersonID
from ...api import FaceRecognizer, FaceRecognizerInput, FaceRecognizerResult, FaceResult
from .config import IouTrackerConfig


@dataclass
class Track:
    person_id: PersonID
    bbox: FaceBoundingBox
    last_seen_frame: int
    lost_count: int = 0
    
    def update(self, bbox: FaceBoundingBox, frame_id: int):
        self.bbox = bbox
        self.last_seen_frame = frame_id
        self.lost_count = 0


class IouTracker(FaceRecognizer):
    def __init__(self, config: IouTrackerConfig):
        self.max_lost_frames = config.max_lost_frames
        self.iou_threshold = config.iou_threshold
        self.max_persons = config.max_persons

    def reset(self):
        self.frame_count = 0
        self.persons: Dict[PersonID, Track] = {}
        self.next_person_id = 1

    def __enter__(self):
        self.reset()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb  # 未使用パラメータ
        self.reset()

    def __call__(self, input: FaceRecognizerInput) -> FaceRecognizerResult:
        start_time = time.time()
        self.frame_count += 1

        if not input.faces:
            return FaceRecognizerResult(results=[], processing_time=time.time() - start_time)

        # 失われたトラックのクリーンアップ
        for person_id in list(self.persons.keys()):
            if self.persons[person_id].lost_count >= self.max_lost_frames:
                del self.persons[person_id]

        # 既存人物とのマッチング
        matches:Dict[int, PersonID] = {}  # face_index -> person_id
        for person_id, track in self.persons.items():
            best_idx, best_iou = None, 0.0
            for i, face in enumerate(input.faces):
                iou = track.bbox.iou(face)
                if iou > self.iou_threshold and iou > best_iou:
                    best_idx, best_iou = i, iou
            
            if best_idx is not None:
                matches[best_idx] = person_id
                track.update(input.faces[best_idx], self.frame_count)
            else:
                track.lost_count += 1

        # 新しい人物の検出
        for i, face in enumerate(input.faces):
            if i not in matches:
                if len(self.persons) < self.max_persons:
                    person_id = PersonID(self.next_person_id)
                    track = Track(person_id, face, self.frame_count)
                    self.persons[person_id] = track
                    matches[i] = person_id
                    self.next_person_id += 1

        results:List[FaceResult] = []
        for face_idx in range(len(input.faces)):
            if face_idx in matches:
                person_id = matches[face_idx]
                results.append(FaceResult(person_id=person_id, confidence=1.0))
            else:
                results.append(FaceResult(person_id=PersonID(-1), confidence=0.0))  # -1 = unknown
        
        return FaceRecognizerResult(results=results, processing_time=time.time() - start_time)