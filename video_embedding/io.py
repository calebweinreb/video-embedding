"""Video reader for self-supervised learning."""
import numpy as np
import cv2
from vidio.read import OpenCVReader

def get_clip(path, start, duration=60):
<<<<<<< HEAD
    """
    Read a video clip from a file using OpenCV.
    
    Args:
        path (str): Path to the video file.
        start (int): Start frame index for the clip.
        duration (int): Duration of the clip in frames.
        
    Returns:
        np.ndarray: Video as array of frames.
    """
=======
    """Read a video clip from a file using OpenCV."""
>>>>>>> 50029a507ff19203422f6efa385453e81d2bbd28
    reader = OpenCVReader(path)
    clip = reader[start : start + duration]
    return np.stack(clip)

