"""Video reader for self-supervised learning."""

import numpy as np
from vidio.read import OpenCVReader


def get_clip(path: str, start: int, duration: int = 60) -> np.ndarray:
    """Read video clip from a file using OpenCV.

    Args:
        path: Path to the video file.
        start: Start frame index for the clip.
        duration: Duration of the clip in frames.

    Returns:
        Video as array of frames.
    """
    reader = OpenCVReader(path)
    clip = reader[start : start + duration]
    return np.stack(clip)
