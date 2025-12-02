from .utils import process_frames


def apply_green(frames):
    processed_data = process_frames(frames)
    BVP = processed_data[:, 1]
    return BVP

