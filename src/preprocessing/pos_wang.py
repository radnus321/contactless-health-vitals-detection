import numpy as np
import math
from scipy import signal
from utils import detrend_signal, process_frames


def apply_pos_wang(frames, fs, win_sec=1.6):
    RGB = process_frames(frames)
    N = RGB.shape[0]
    H = np.zeros((N,))
    l = math.ceil(win_sec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = RGB[m:n, :] / np.mean(RGB[m:n, :], axis=0)
            S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn.T
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            h = h - np.mean(h)
            H[m:n] += h

    BVP = detrend_signal(H, lambda_val=100)

    b, a = signal.butter(1, [0.75/fs*2, 3/fs*2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP
