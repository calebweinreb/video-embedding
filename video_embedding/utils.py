"""Transformation functions for video data."""

import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter
import random
from albumentations.pytorch import ToTensorV2
from albumentations import ReplayCompose
from vidio.read import OpenCVReader
import albumentations as A


def transform_video(video_array):
    """
    - Normalize from 0-255 to 0-1
    - Standardize channels using hard-coded mean and std
    - Change channel order
    - Convert to tensor

    Args:
        video_array (numpy.ndarray): 4D or 5D array with shape ([B], T, H, W, C), where B is the
        batch size, T is the number of frames, H is height, W is width, and C is channels (RGB).

    Returns:
        torch.Tensor: Transformed video tensor with shape ([B], C, T, H, W)
    """
    video_array = video_array.astype(np.float32) / 255

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    video_array = (video_array - mean) / std

    if len(video_array.shape) == 4:
        video_array = np.transpose(video_array, (3, 0, 1, 2))
    else:
        video_array = np.transpose(video_array, (0, 4, 1, 2, 3))
    return torch.from_numpy(video_array)


def untransform_video(video_tensor):
    """
    Inverts the transformations applied by the `transform_video` function.

    Args:
        video_tensor (torch.Tensor): Transformed video tensor with shape ([B], C, T, H, W), where B
        is the batch size, C is channels (RGB), T is the frame count, H is height, and W is width.

    Returns:
        numpy.ndarray: 4D or 5D array of shape ([B], T, H, W, C) representing the original video(s).
    """
    video_array = video_tensor.numpy()

    if len(video_array.shape) == 4:
        video_array = np.transpose(video_array, (1, 2, 3, 0))
    else:
        video_array = np.transpose(video_array, (0, 2, 3, 4, 1))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    video_array = (video_array * std) + mean
    video_array = (video_array * 255).astype(np.uint8)
    return video_array
