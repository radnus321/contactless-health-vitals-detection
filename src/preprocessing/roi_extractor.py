import yaml
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


class ROIExtractor:
    def __init__(self, roi_type="forehead"):
        self.roi_type = roi_type
        self.roi_size = cfg["roi_extraction"]["roi_size"]
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmark_map = {
                "forehead": cfg["roi_extraction"]["forehead_landmarks"],
                "left_cheek": cfg["roi_extraction"]["left_cheek_landmarks"],
                "right_cheek": cfg["roi_extraction"]["right_cheek_landmarks"]
        }

    def detect_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0].landmark

    def get_roi_coords(self, landmarks, shape):
        h, w, _ = shape

        ids = self.landmark_map[self.roi_type]

        xs, ys = [], []
        for idx in ids:
            xs.append(landmarks[idx].x * w)
            ys.append(landmarks[idx].y * h)

        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))

        half = self.roi_size // 2

        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) < self.roi_size:
            if x1 == 0:
                x2 = min(self.roi_size, w)
            else:
                x1 = max(w - self.roi_size, 0)

        if (y2 - y1) < self.roi_size:
            if y1 == 0:
                y2 = min(self.roi_size, h)
            else:
                y1 = max(h - self.roi_size, 0)

        return int(x1), int(y1), int(x2), int(y2)

    def extract_roi(self, frame):
        landmarks = self.detect_landmarks(frame)
        if landmarks is None:
            return None
        x1, y1, x2, y2 = self.get_roi_coords(landmarks, frame.shape)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def release(self):
        self.face_mesh.close()
