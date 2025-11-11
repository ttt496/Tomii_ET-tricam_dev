import time
import numpy as np
import torch
from typing import Dict, List, Optional, Sequence, Set
from dataclasses import dataclass

from core.dtypes.person import PersonID
from core.dtypes.media import ImageArray
from core.integrators.face_detector.dtypes.bounding import FaceBoundingBox
from core.models.mp_facemesh.model import MpFacemeshModel
from core.models.lstm_base_01.model import LSTMBase01Model
from ...api import BlinkDetector, EARBlinkDetails, BlinkDetectorInput, BlinkDetectorResult, BlinkResult
from .config import LstmBase01BlinkDetectorConfig
from .util import (
    extract_eye_bb, 
    # validate_eye_position
)

@dataclass
class PersonTrackingState:
    h: torch.Tensor
    c: torch.Tensor
    frames_since_last_seen: int = 0
    detection_count: int = 0

class LstmHiddenPersonManager:
    def __enter__(self, model:LSTMBase01Model):
        self.person_states: Dict[PersonID, PersonTrackingState] = {}
        self.model = model
        self.hidden_size = model.hidden_size
        self.device = model.device
        self.num_layers = model.num_layers
        self.bidirectional = model.bidirectional

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model = None
        self.person_states = {}

    def update_person_states(self, person_ids:Set[PersonID]):
        assert self.model

        delete_person_ids = set()
        for person_id in self.person_states:
            state = self.person_states[person_id]
            if person_id in person_ids: 
                state.frames_since_last_seen = 0
                state.detection_count += 1
            else:
                state.frames_since_last_seen += 1
                threshold = self.model.config.person_retention_frames + state.detection_count
                if state.frames_since_last_seen > threshold:
                    delete_person_ids.add(person_id)
        for person_id in delete_person_ids: del self.person_states[person_id]
        
        def create_zero_vector():
            num_directions = 2 if self.bidirectional else 1
            h = torch.zeros(self.num_layers * num_directions, 1, self.hidden_size, device=self.device)
            c = torch.zeros(self.num_layers * num_directions, 1, self.hidden_size, device=self.device)
            return h, c

        for person_id in person_ids:
            if person_id in self.person_states: continue
            h, c = create_zero_vector()
            self.person_states[person_id] = PersonTrackingState(h, c, 0, 1)

    def create_inputs(self, ear_results:Sequence[EARBlinkDetails]):
        assert self.model
        input_data, hidden_data, cell_data = [], [], []
        ear_result_map = {er.person_id:er for er in ear_results}
        for person_id, state in self.person_states.items():
            er = ear_result_map[person_id] if person_id in ear_result_map else None
            input_data.append([[
                er.left_eye_bb.ear if er and er.left_eye_bb is not None else -1.0, 
                er.right_eye_bb.ear if er and er.right_eye_bb is not None else -1.0
            ]])
            hidden_data.append(state.h)
            cell_data.append(state.c)
        input_batch = torch.Tensor(input_data).to(self.model.device)
        hidden_batch = torch.cat(hidden_data, dim=1).to(self.model.device)
        cell_batch = torch.cat(cell_data, dim=1).to(self.model.device)
        return input_batch, hidden_batch, cell_batch

    def update_hidden(self, new_h:torch.Tensor, new_c:torch.Tensor):
        for i, person_id in enumerate(self.person_states):
            self.person_states[person_id].h = new_h[:, i:i+1, :]
            self.person_states[person_id].c = new_c[:, i:i+1, :]

    def filter_output(self, output:np.ndarray, person_ids:List[PersonID]) -> np.ndarray:
        dic = {person_id: i for i, person_id in enumerate(self.person_states.keys())}
        indeces = [dic[person_id] for person_id in person_ids]
        return output[indeces]

    def __call__(self, ear_results:Sequence[EARBlinkDetails]) -> np.ndarray:
        assert len(ear_results) > 0
        assert self.model

        self.update_person_states(set(map(lambda er:er.person_id, ear_results)))
        input_batch, hidden_batch, cell_batch = self.create_inputs(ear_results)

        with torch.no_grad():
            logits, new_h, new_c = self.model(input_batch, hidden_batch, cell_batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        assert probs.shape[1] == 3  # [open, half-open, closed]
        self.update_hidden(new_h, new_c)
        filtered_probs = self.filter_output(probs, list(map(lambda er:er.person_id, ear_results)))
        blink_probs = filtered_probs[:,1] + filtered_probs[:,2]
        return blink_probs

    def get_sequence_length(self, person_id:PersonID)->int:
        state = self.person_states.get(person_id, None)
        return state.detection_count if state else 0


class LstmBase01BlinkDetector(BlinkDetector):
    def __init__(self, config: LstmBase01BlinkDetectorConfig):
        self.lstm_config = config.lstm_base_01
        self.facemesh_config = config.mp_facemesh
        
        self.facemesh_model:MpFacemeshModel = MpFacemeshModel(self.facemesh_config)
        self.lstm_model:Optional[LSTMBase01Model] = None
        self.lstm_hidden_manager = LstmHiddenPersonManager()
    
    def __enter__(self):
        self.facemesh_model.__enter__()
        self.lstm_model = LSTMBase01Model.load_model(self.lstm_config.model_path, self.lstm_config)
        self.lstm_hidden_manager.__enter__(self.lstm_model)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.facemesh_model.__exit__(exc_type, exc_val, exc_tb)
        if self.lstm_model:
            self.lstm_model = None
        self.lstm_hidden_manager.__exit__(exc_type, exc_val, exc_tb)

    def process_mediapipe(self, image: ImageArray, faces: Sequence[FaceBoundingBox], person_ids:Optional[List[PersonID]]) -> Sequence[EARBlinkDetails]:
        assert self.facemesh_model
        ear_results:List[EARBlinkDetails] = []

        person_ids = person_ids or list(range(len(faces)))
        for person_id, face_bbox in zip(person_ids, faces):
            x1, y1 = int(face_bbox.x), int(face_bbox.y)
            x2, y2 = x1 + int(face_bbox.width), y1 + int(face_bbox.height)
            face_image = image[y1:y2, x1:x2]
            
            # Convert BGR to RGB if needed
            face_rgb = face_image[:, :, ::-1] if len(face_image.shape) == 3 and face_image.shape[2] == 3 else face_image
            landmarks = self.facemesh_model(face_rgb)
            
            if not landmarks: 
                ear_results.append(EARBlinkDetails(person_id=person_id, left_eye_bb=None, right_eye_bb=None))
                continue
            left_eye_bb = extract_eye_bb(landmarks, 'left')
            right_eye_bb = extract_eye_bb(landmarks, 'right')
            
            # if isinstance(face_bbox, FaceEyeBoundingBox):
            #     if face_bbox.left_eye:
            #         left_eye_bb = left_eye_bb if validate_eye_position(left_eye_bb, face_bbox.left_eye) else None
            #     if face_bbox.right_eye:
            #         right_eye_bb = right_eye_bb if validate_eye_position(right_eye_bb, face_bbox.right_eye) else None
            
            ear_results.append(EARBlinkDetails(
                person_id=person_id,
                left_eye_bb=left_eye_bb,
                right_eye_bb=right_eye_bb,
            ))
        return ear_results

    def process_lstm(self, ear_results:Sequence[EARBlinkDetails])->Sequence[BlinkResult]:
        assert self.lstm_model
        if len(ear_results) == 0: return []
            
        blink_probs = self.lstm_hidden_manager(ear_results)

        results:List[BlinkResult] = []
        for i, ear_result in enumerate(ear_results):
            sequence_length = self.lstm_hidden_manager.get_sequence_length(ear_result.person_id)
            results.append(BlinkResult(
                blink_probability=blink_probs[i] if sequence_length >= 5 else None,
                person_id=ear_result.person_id,
                ear_result=ear_result
            ))
        return results

    def __call__(self, input: BlinkDetectorInput) -> BlinkDetectorResult:        
        start_time = time.time()

        ear_results = self.process_mediapipe(input.image, input.faces, input.person_ids)
        blink_results = self.process_lstm(ear_results)
        
        processing_time = time.time() - start_time
        return BlinkDetectorResult(results=blink_results, processing_time=processing_time)