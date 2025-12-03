import yaml
import numpy as np
from pathlib import Path
from .roi_extractor import ROIExtractor
from .pos_wang import apply_pos_wang
from .chrome_dehaan import apply_chrome_dehaan
from .green import apply_green

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


class SignalExtractor:
    def __init__(self):
        self.roi_list = None
        self.pos_wang_signal = None
        self.chrome_dehaan_signal = None
        self.green_signal = None

    def extract_rois(self, video_list):
        forehead_roi_extractor = ROIExtractor("forehead")
        left_cheek_roi_extractor = ROIExtractor("left_cheek")
        right_cheek_roi_extractor = ROIExtractor("right_cheek")
        roi_list = []
        for video in video_list:
            rois = []
            for frame in video:
                forehead_roi = forehead_roi_extractor.extract_roi(frame)
                left_cheek_roi = left_cheek_roi_extractor.extract_roi(frame)
                right_cheek_roi = right_cheek_roi_extractor.extract_roi(frame)
                rois.append([forehead_roi, left_cheek_roi, right_cheek_roi])
            roi_list.append(rois)
        roi_array = np.array(roi_list)
        roi_array = roi_array.transpose(0, 2, 1, 3, 4, 5)
        self.roi_list = roi_array

    def extract_signals(self, video_list, fs):
        self.extract_rois(video_list)
        N, R, F, H, W, C = self.roi_list.shape
        methods = cfg["signal_processing"]["methods"]

        pos_wang_all = []
        chrome_all = []
        green_all = []

        for vid_idx in range(N):
            pos_video = []
            chrome_video = []
            green_video = []
            for roi_idx in range(R):
                frames = self.roi_list[vid_idx, roi_idx]   # (F, 64, 64, 3)
                if "pos_wang" in methods:
                    pos_video.append(apply_pos_wang(frames, fs))
                if "chrome_dehaan" in methods:
                    chrome_video.append(apply_chrome_dehaan(frames, fs))
                if "green" in methods:
                    green_video.append(apply_green(frames))
            pos_wang_all.append(pos_video)
            chrome_all.append(chrome_video)
            green_all.append(green_video)

        self.pos_wang_signal = np.array(pos_wang_all)
        self.chrome_dehaan_signal = np.array(chrome_all)
        self.green_signal = np.array(green_all)
