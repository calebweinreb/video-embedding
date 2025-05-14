"""Video preprocessing for augmentation."""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter
import random
from albumentations.pytorch import ToTensorV2
from albumentations import ReplayCompose
from vidio.read import OpenCVReader
import albumentations as A

def generate_trajectory(duration, dof, gaussian_kernel, multiplier):
    """
    Create a smooth two-dimensional random trajectory using a t-distribution.

    Args:
        duration (int): Number of time steps in the trajectory.
        dof (float): Degrees of freedom for the t-distribution.
        gaussian_kernel (int): Smoothing kernel size.
        multiplier (float): Scaling factor for the trajectory magnitude.

    Returns:
        np.ndarray: Integer-valued 2D trajectory of shape (duration, 2).
    """
    trajectory = np.random.standard_t(dof, size=(duration,2))
    trajectory = gaussian_filter1d(trajectory, gaussian_kernel, axis=0)
    trajectory = trajectory - trajectory.mean(0)
    return (trajectory * multiplier).astype(int)

def translate(image, shift_x, shift_y):
    """
    Apply an affine transformation to shift an image by (shift_x, shift_y).

    Args:
        image (np.ndarray): input image, 
        shift_x (int): horizontal shift, 
        shift_y (int): vertical shift.

    Returns: 
        np.ndarray: translated image.
    """
    h, w = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return translated_image