import os
import cv2
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


class DatasetLoader:
    def __init__(self, root_dir: str, metadata_path: str, extensions: Tuple[str] = (".mp4", ".avi", ".mov")):
        self.root_dir = root_dir
        self.metadata_path = metadata_path
        self.extensions = extensions
        self.target_list = None
        self.video_list = None

    def load_metadata(self, sheet_name: str):
        try:
            return pd.read_excel(self.metadata_path, sheet_name=sheet_name)
        except ValueError:
            try:
                # trying with a leading space because of incorrect entry
                return pd.read_excel(self.metadata_path, sheet_name=" " + sheet_name)
            except ValueError:
                raise ValueError(f"Worksheet '{sheet_name}' or ' {sheet_name}' not found.")

    def load_videos_paths(self) -> List[str]:
        video_files = []
        for d in os.listdir(self.root_dir):
            dir_path = os.path.join(self.root_dir, d)
            if not os.path.isdir(dir_path):
                continue
            dir_files = []
            for video in os.listdir(dir_path):
                video_path = os.path.join(dir_path, video)
                video_name, _ = os.path.splitext(video)
                dir_files.append((video_name, video_path))
            video_files.append((d, dir_files))
        return video_files

    def load_video(self, path: str, num_frames: Optional[int]) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        frames = []
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1
            if num_frames is not None and count >= num_frames:
                break
        cap.release()
        return np.asarray(frames, dtype=np.uint8)

    def create_dataset(self, num_frames: Optional[int]):
        video_list = []
        target_list = []

        file_paths = self.load_videos_paths()

        for (dir_name, dir_files) in file_paths:
            try:
                dir_metadata = self.load_metadata(dir_name)
            except Exception:
                continue
            for (video_name, video_path) in dir_files:
                video = self.load_video(path=video_path, num_frames=num_frames)
                if video is None:
                    continue

                rows = dir_metadata[dir_metadata["Video File Name"] == video_name]
                if rows.empty:
                    continue
                try:
                    oxygen = rows["Oxygen Level"].iloc[0]
                except Exception:
                    continue

                video_list.append(video)
                target_list.append(oxygen)
        self.video_list = np.array(video_list)
        self.target_list = np.array(target_list)
