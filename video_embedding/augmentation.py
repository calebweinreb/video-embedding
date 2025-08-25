import numpy as np
import albumentations as A
from scipy.ndimage import gaussian_filter1d
import cv2
from .utils import crop_image
from typing import Tuple, Union
from abc import ABC, abstractmethod


class VideoAugmentation(ABC):
    """Base class for any video‐level augmentation.

    Subclasses must implement the `_augment` method to perform
    augmentation on a 4-D video array.
    """

    def __call__(self, video_array: np.ndarray) -> np.ndarray:
        """Apply augmentation and validate output shape and dtype.

        Args:
            video_array: Input video of shape ``(T, H, W, C)``.

        Returns:
            Augmented video array of same shape and dtype.

        Raises:
            ValueError: If the output is not a 4-D array or dtype mismatch.
        """
        augmented = self._augment(video_array)
        if not isinstance(augmented, np.ndarray):
            raise ValueError(f"{self.__class__.__name__} must return an ndarray.")
        if augmented.ndim != 4:
            raise ValueError(f"{self.__class__.__name__} must return a 4-D array.")
        if augmented.dtype != video_array.dtype:
            raise ValueError(
                f"{self.__class__.__name__} must return array with dtype {video_array.dtype}."
            )
        return augmented

    @abstractmethod
    def _augment(self, video_array: np.ndarray) -> np.ndarray:
        """Perform augmentation on the video array."""
        ...


class TemporalCrop(VideoAugmentation):
    """Randomly crop video along temporal axis to a fixed frame count."""

    def __init__(self, target_duration):
        """
        Args:
            target_duration: Number of frames to keep after cropping.
        """
        self.target_duration = target_duration

    def _augment(self, video_array: np.ndarray) -> np.ndarray:
        """Crop video to `target_duration` frames."""
        num_frames = video_array.shape[0]
        if num_frames < self.target_duration:
            raise ValueError(
                f"Video too short ({num_frames} frames) for TemporalCrop("
                f"{self.target_duration})."
            )
        if num_frames == self.target_duration:
            return video_array
        start = np.random.randint(0, num_frames - self.target_duration + 1)
        return video_array[start : start + self.target_duration]


class TranslationDrift(VideoAugmentation):
    """Apply random camera drift to each frame with given probability."""

    def __init__(
        self,
        p=0.9,
        dof=1.5,
        gaussian_kernel=15,
        multiplier=6,
    ):
        """
        Args:
            p: Probability of applying drift.
            dof: Degrees of freedom for the t-distribution.
            gaussian_kernel: Size of the Gaussian kernel for smoothing.
            multiplier: Scaling factor for the trajectory magnitude.
        """
        self.p = p
        self.dof = dof
        self.gaussian_kernel = gaussian_kernel
        self.multiplier = multiplier

    @staticmethod
    def generate_trajectory(
        duration: int, dof: float, gaussian_kernel: int, multiplier: float
    ) -> np.ndarray:
        """Generate a random trajectory for camera drift.

        Args:
            duration: Number of frames in the video.
            dof: Degrees of freedom for the t-distribution.
            gaussian_kernel: Size of the Gaussian kernel for smoothing.
            multiplier: Scaling factor for the trajectory magnitude.

        Returns:
            Random trajectory of shape ``(duration, 2)``.
        """
        traj = np.random.standard_t(dof, size=(duration, 2))
        traj = gaussian_filter1d(traj, gaussian_kernel, axis=0)
        traj = traj - traj.mean(0)
        return (traj * multiplier).astype(int)

    @staticmethod
    def translate(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """Translate an image by a given x and y shift."""
        h, w = image.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _augment(self, video_array: np.ndarray) -> np.ndarray:
        """Apply drift to each frame with probability `p`."""
        if np.random.uniform() < self.p:
            duration = video_array.shape[0]
            traj = self.generate_trajectory(
                duration, self.dof, self.gaussian_kernel, self.multiplier
            )
            return np.stack(
                [
                    self.translate(frame, dx, dy)
                    for frame, (dx, dy) in zip(video_array, traj)
                ]
            )
        return video_array


class CenterCrop(VideoAugmentation):
    """Spatially center-crop each frame in the video."""

    def __init__(
        self,
        crop_size: int,
        border_mode: int = cv2.BORDER_REFLECT,
        border_value: Union[int, Tuple[int, int, int]] = 0,
    ):
        """
        Args:
            crop_size: Size of the square crop to apply.
            border_mode: OpenCV border mode for padding (default is cv2.BORDER_REFLECT).
            border_value: Value for BORDER_CONSTANT; scalar or tuple for multi-channel images.

        """
        self.crop_size = crop_size
        self.border_mode = border_mode
        self.border_value = border_value

    def _augment(self, video_array: np.ndarray) -> np.ndarray:
        """Crop video frames around their centroid."""
        centroid = (video_array.shape[2] // 2, video_array.shape[1] // 2)
        return np.stack(
            [
                crop_image(
                    frame, centroid, self.crop_size, self.border_mode, self.border_value
                )
                for frame in video_array
            ]
        )


class AlbumentationsAugs(VideoAugmentation):
    """Wrap Albumentations transforms for consistent video augmentation."""

    def __init__(self, alb_transforms):
        """
        Args:
            alb_transforms: List of Albumentations transform instances.
        """
        self.transform = A.ReplayCompose(alb_transforms)

    @classmethod
    def default(cls):
        """Create an instance with the default Albumentations pipeline."""
        transforms = [
            A.HorizontalFlip(p=0.5),  # Horizontal flip
            A.VerticalFlip(p=0.5),  # Vertical flip
            A.Affine(
                translate_percent=0.05,
                scale=(0.8, 1.1),
                rotate=(-360, 360),
                p=0.8,
                border_mode=cv2.BORDER_REFLECT,
            ),
            A.RandomBrightnessContrast(p=0.5),  # Brightness & contrast
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),  # Gaussian blur
            A.HueSaturationValue(p=0.5, hue_shift_limit=5),  # HSV shift
            A.RandomGamma(p=0.3),  # Gamma adjustment
            A.ColorJitter(p=0.3),  # Color jittering
            A.MotionBlur(blur_limit=7, p=0.2),  # Motion blur
        ]
        return cls(transforms)

    def _augment(self, video_array: np.ndarray) -> np.ndarray:
        """Apply ReplayCompose and replay transforms across all frames."""
        augmented_video = np.zeros_like(video_array)
        first = video_array[0]
        augmented = self.transform(image=first)
        augmented_video[0] = augmented["image"]
        replay_data = augmented["replay"]
        for i in range(1, video_array.shape[0]):
            frame = video_array[i]
            result = A.ReplayCompose.replay(replay_data, image=frame)
            augmented_video[i] = result["image"]
        return augmented_video


class VideoAugmentator:
    """Compose multiple video augmentations into a single pipeline."""

    def __init__(self, augmentations):
        """
        Args:
            augmentations: Sequence of VideoAugmentation instances to apply in order.
        """
        self.augmentations = list(augmentations)

    def __call__(self, video_array: np.ndarray) -> np.ndarray:
        """Apply each augmentation in sequence to the video."""
        for aug in self.augmentations:
            video_array = aug(video_array)
        return video_array
