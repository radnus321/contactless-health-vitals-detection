import numpy as np
from scipy import sparse


def detrend_signal(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def process_frames(frames):
    RGB = []
    for frame in frames:
        mean_rgb = np.mean(frame, axis=(0,1))
        RGB.append(mean_rgb)
    return np.asarray(RGB)
