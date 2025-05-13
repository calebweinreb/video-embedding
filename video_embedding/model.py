"""Necessities for model implementation."""

def transform_video(video_array):
    """
    - Normalize from 0-255 to 0-1
    - Standardize channels using mean and std from Kinetics
    - Change channel order
    - Convert to tensor
    
    Args:
    video_array (numpy.ndarray): 4D array with shape ([B], T, H, W, C), where
        T is the number of frames, H is height, W is width, and C is channels (assumed to be RGB).
    
    Returns:
    torch.Tensor: Transformed video tensor with shape ([B], C, T, H, W)
    """
    video_array = video_array.astype(np.float32)/255

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    video_array = (video_array - mean) / std

    if len(video_array.shape)==4:
        video_array = np.transpose(video_array, (3, 0, 1, 2))
    else:
        video_array = np.transpose(video_array, (0, 4, 1, 2, 3))

    return torch.from_numpy(video_array)

def untransform_video(video_tensor):
    """
    Inverts the transformations applied by the `transform_video` function.

    Args:
    video_tensor (torch.Tensor): Transformed video tensor with shape ([B], C, T, H, W)

    Returns:
    numpy.ndarray: 4D array with shape ([B], T, H, W, C) representing the original video(s).
    """
    video_array = video_tensor.numpy()

    if len(video_array.shape)==4:
        video_array = np.transpose(video_array, (1, 2, 3, 0))
    else:
        video_array = np.transpose(video_array, (0, 2, 3, 4, 1))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    video_array = (video_array * std) + mean


    video_array = (video_array * 255).astype(np.uint8)

    return video_array
    
class VideoAugmentator():
    """Create class for applying consistent augmentations to a video sequence using albumentations."""
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

    """
    Apply temporal crop, drift, augmentations, and spatial crop.

    Returns:
        np.ndarray: Augmented video."""

        video_array = random_temporal_crop(video_array, self.duration)
        video_array = random_drift(video_array, *self.drift_params)
        video_array = apply_albumentations_to_video(video_array, self.transform)
        video_array = center_crop(video_array, self.crop_size)
        return video_array