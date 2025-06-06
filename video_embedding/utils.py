import numpy as np
import torch
from typing import List, Tuple, Union, Optional
from vidio.read import OpenCVReader
import imageio
import tqdm


def transform_video(video_array: np.ndarray) -> torch.Tensor:
    """Normalize video clip and reformat as torch tensor

    Args:
        video_array: 4D or 5D array of video frames with shape ``([B], T, H, W, C)``.

    Returns:
        Transformed video tensor with shape ``([B], C, T, H, W)``.
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


def untransform_video(video_tensor: torch.Tensor) -> np.ndarray:
    """Invert the transformations applied by the `transform_video` function.

    Args:
        video_tensor: Transformed video tensor with shape ``([B], C, T, H, W)``.

    Returns:
        4D or 5D array of shape ``([B], T, H, W, C)`` representing the original video(s).
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


def sample_timepoints(
    video_paths: List[str],
    num_samples: int,
    video_lengths: Optional[List[int]] = None,
    clip_size: int = 1,
) -> List[Tuple[str, int]]:
    """Uniformly sample frame indexes from an ensemble of videos.

    Args:
        video_paths: List of video file paths.
        num_samples: Number of timepoints to sample.
        video_lengths: Video lengths in frames. If ``None``, lengths are determined from files.
        clip_size: Ensure samples are at least this distance from the end of the video.

    Returns:
        List of tuples ``(video_path, timepoint)``.
    """
    if video_lengths is None:
        video_lengths = [len(OpenCVReader(p)) for p in video_paths]

    p = np.array(video_lengths) + 1 - clip_size
    video_probabilities = p / np.sum(p)

    video_indexes = np.random.choice(len(video_paths), size=num_samples, p=video_probabilities)
    timepoints = [np.random.randint(0, video_lengths[i] - clip_size + 1) for i in video_indexes]
    return [(video_paths[i], t) for i, t in zip(video_indexes, timepoints)]


def crop_video(
    video_path: str,
    cropped_video_path: str,
    track: np.ndarray,
    crop_size: Union[int, Tuple[int, int]],
    quality: int = 5,
    constrain_track: Optional[bool] = True,
) -> None:
    """Crop a video around a time-varying centroid.

    Args:
        video_path: Path to the input video.
        cropped_video_path: Path to save the cropped video.
        track: Array of shape ``(frames, 2)`` containing the crop centroid ``(x, y)`` for each frame.
        crop_size: Size of the crop. If an integer is provided, it crops a square of that size.
        quality: Quality of the output video passed to ``imageio.get_writer``.
        constrain_track: If ``True``, ensures the cropped area does not exceed the video boundaries.
    """
    reader = OpenCVReader(video_path)

    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)

    if constrain_track:
        h, w = reader[0].shape[:2]
        track[:, 0] = np.clip(track[:, 0], crop_size[0] // 2, w - crop_size[0] // 2)
        track[:, 1] = np.clip(track[:, 1], crop_size[1] // 2, h - crop_size[1] // 2)

    with imageio.get_writer(
        cropped_video_path, pixelformat="yuv420p", fps=reader.fps, quality=quality
    ) as writer:
        for frame_ix in tqdm.trange(len(reader)):
            frame = reader[frame_ix]
            cen = track[frame_ix]
            cropped_frame = crop_image(frame, cen, crop_size)
            writer.append_data(cropped_frame)
