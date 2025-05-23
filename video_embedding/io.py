"""Video reader for self-supervised learning."""
import numpy as np
import cv2
from vidio.read import OpenCVReader
import os, glob
from vidio.read import OpenCVReader
from typing import List, Tuple
import pathlib


def get_clip(path, start, duration=60):
    """
    Read video clip from a file using OpenCV.
    
    Args:
        path (str): Path to the video file.
        start (int): Start frame index for the clip.
        duration (int): Duration of the clip in frames.
        
    Returns:
        np.ndarray: Video as array of frames.
    """
    reader = OpenCVReader(path)
    clip = reader[start : start + duration]
    return np.stack(clip)
