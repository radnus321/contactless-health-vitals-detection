from utils import process_frames


def apply_green(frames):
    precessed_data = process_frames(frames)
    BVP = precessed_data[:, 1, :]
    BVP = BVP.reshape(-1)
    return BVP

