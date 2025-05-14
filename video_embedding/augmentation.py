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
    Crop a video around its center to a fixed size.

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
    Augment a video with random camera drift.
    
    Args:
        video_array (np.ndarray): Input video, 
        drift_prob (float): Probability of applying drift.
        dof (float): Degrees of freedom for the t-distribution.
        gaussian_kernel (int): Smoothing kernel size.
        multiplier (float): Scaling factor for the trajectory magnitude.
    
    Returns:
        np.ndarray: Augmented video with random drift.
    """
    if np.random.uniform() < drift_prob:
        duration = video_array.shape[0]
        trajectory = generate_trajectory(duration, dof, gaussian_kernel, multiplier)
        return np.stack([translate(im, *xy) for im,xy in zip(video_array, trajectory)])
    else:
        return video_array
        

def generate_trajectory(duration, dof, gaussian_kernel, multiplier):
    """
    Create a smooth two-dimensional random trajectory (for random camera drift).

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


class VideoAugmentator():
    """Applies consistent augmentations to a video sequence using albumentations."""
    def __init__(
        self,
        duration=30,
        crop_size=256, 
        drift_prob=0.9,
        gaussian_kernel=15, 
        multiplier=10, 
        dof=1.5,   
    ):
        self.duration = duration
        self.crop_size = crop_size
        self.drift_params = (drift_prob, dof, gaussian_kernel, multiplier)
        
        self.transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),                 # Horizontal flip
            A.VerticalFlip(p=0.5),                   # Vertical flip
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.2,0.1), rotate_limit=360, p=0.8, border_mode = cv2.BORDER_REFLECT),
            A.RandomBrightnessContrast(p=0.5),       # Adjust brightness and contrast
            A.GaussianBlur(blur_limit=(1, 3), p=0.3), # Apply Gaussian blur
            A.HueSaturationValue(p=0.5, hue_shift_limit=5),             # Random hue, saturation, value shifts
            A.RandomGamma(p=0.3),                    # Random gamma adjustment
            A.ColorJitter(p=0.3),                    # Color jittering
            A.MotionBlur(blur_limit=7, p=0.2),       # Motion blur
        ])        

    def __call__(self, video_array):
        video_array = random_temporal_crop(video_array, self.duration)
        video_array = random_drift(video_array, *self.drift_params)
        video_array = apply_albumentations_to_video(video_array, self.transform)
        video_array = center_crop(video_array, self.crop_size)
        return video_array