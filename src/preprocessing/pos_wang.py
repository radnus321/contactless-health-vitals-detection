import numpy as np
import yaml
import math
from scipy import signal
from pathlib import Path
from .utils import detrend_signal, process_frames

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


def apply_pos_wang(frames, fs):
    RGB = process_frames(frames)
    N = RGB.shape[0]
    H = np.zeros((N,))
    win_sec = cfg["signal_processing"]["pos"]["window_sec"]
    lambda_val = cfg["signal_processing"]["pos"]["detrend_lambda"]
    l = math.ceil(win_sec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = RGB[m:n, :] / np.mean(RGB[m:n, :], axis=0)
            S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn.T
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            h = h - np.mean(h)
            H[m:n] += h

    BVP = detrend_signal(H, lambda_value=lambda_val)

    [lower, upper]= cfg["signal_processing"]["pos"]["bandpass"]

    b, a = signal.butter(1, [lower/fs*2, upper/fs*2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP
