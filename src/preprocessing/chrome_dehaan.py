import numpy as np
import yaml
import math
from scipy import signal
from scipy.signal import resample
from pathlib import Path
from .utils import process_frames

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


def apply_chrome_dehaan(frames, FS):
    RGB = process_frames(frames)
    FN = RGB.shape[0]
    win_sec = cfg["signal_processing"]["chrome"]["window_sec"]
    order = cfg["signal_processing"]["chrome"]["order"]
    [lower, upper] = cfg["signal_processing"]["chrome"]["bandpass"]
    FN = RGB.shape[0]
    NyquistF = 1/2*FS
    B, A = signal.butter(order, [lower/NyquistF, upper/NyquistF], 'bandpass')

    WinL = math.ceil(win_sec*FS)
    if(WinL % 2):
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, signal.windows.hann(WinL))

        temp = SWin[:int(WinL//2)]
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    if len(S) != FN:
        S = resample(S, FN)
    BVP = S
    return BVP
