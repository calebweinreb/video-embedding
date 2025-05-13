"""Video augmentation for self-supervised learning."""
import numpy as np
import albumentations as A
from scipy.ndimage import gaussian_filter1d, median_filter
from albumentations.pytorch import ReplayCompose
import cv2


def apply_albumentations_to_video(video_array, alb_transform):
    """
    Implement albumentations ReplayCompose transformation across all frames in video sequence.
  
    Args:
        video_array (np.ndarray): Video as array of frames.
        alb_transform (ReplayCompose): Albumentations transform with replay capability.

    Returns:
        np.ndarray: Augmented video array.
    """
    augmented_video = np.zeros_like(video_array)
    first_frame = video_array[0]
    augmented = alb_transform(image=first_frame)
    augmented_video[0] = augmented['image']
    replay_data = augmented['replay']
    for i in range(1, video_array.shape[0]):
        frame = video_array[i]
        augmented = A.ReplayCompose.replay(replay_data, image=frame)
        augmented_video[i] = augmented['image']
    return augmented_video


def center_crop(video_array, crop_size):
    """
    Perform center crop on each frame of the video.

    Parameters:
        video_array : np.ndarray
        An array of shape (T, H, W), where:
          - T is the number of frames,
          - H is the frame height,
          - W is the frame width.
    crop_size : int
        The target size (in pixels) for both height and width after cropping.

    Returns: 
        np.ndarray: Center-cropped video.
    """
    h,w = video_array.shape[1:3]
    if h > crop_size:
        video_array = video_array[:,(h-crop_size)//2: -(h-crop_size)//2]
    if w > crop_size:
        video_array = video_array[:,:,(w-crop_size)//2: -(w-crop_size)//2]   
    return video_array


def random_temporal_crop(video_array, duration):
    """
    Crop video randomly along temporal axis to a fixed frame count.

    Args: 
        video_array (np.ndarray): Input video, 
        duration (int): Target number of frames.

    Returns:
        np.ndarray: Cropped video.
    """
    if len(video_array) > duration:
        start = np.random.randint(len(video_array)-duration)
        video_array = video_array[start : start+duration]
    return video_array


def random_drift(video_array, drift_prob, dof, gaussian_kernel, multiplier):
    """
    Introduce random camera drift to each video frame based on a given probability.
    
    Args:
    

    Returns:
    """
    if np.random.uniform() < drift_prob:
        duration = video_array.shape[0]
        trajectory = generate_trajectory(duration, dof, gaussian_kernel, multiplier)
        return np.stack([translate(im, *xy) for im,xy in zip(video_array, trajectory)])
    else:
        return video_array
        

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
        shift_y (int): vertical shift

    Returns: 
        np.ndarray: translated image
    """
    h, w = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return translated_image