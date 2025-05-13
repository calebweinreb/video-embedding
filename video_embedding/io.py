"""Video reader for self-supervised learning."""
import numpy as np
import cv2
from vidio.read import OpenCVReader

def get_clip(path, start, duration=60):
    """
    Read a video clip from a file using OpenCV."""
    reader = OpenCVReader(path)
    clip = reader[start : start + duration]
    return np.stack(clip)

