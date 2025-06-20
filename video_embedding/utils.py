import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Union, Optional, Iterator
from vidio.read import OpenCVReader
import imageio
import tqdm
import h5py
import os
import json
import re
import glob
from collections import deque


def transform_video(video_array: np.ndarray) -> torch.Tensor:
    """Normalize video clip, permute dimensions, and convert to tensor.

    Args:
        video_array: 4D or 5D array of video frames with shape ([B], T, H, W, C).

    Returns:
        4D or 5D [0-1] normalized tensor with shape ([B], C, T, H, W).
    """
    video_array = video_array.astype(np.float32) / 255.0
    if video_array.ndim == 4:
        video_array = np.transpose(video_array, (3, 0, 1, 2))  # (C, T, H, W)
    elif video_array.ndim == 5:
        video_array = np.transpose(video_array, (0, 4, 1, 2, 3))  # (B, C, T, H, W)
    video_tensor = torch.from_numpy(video_array)
    return video_tensor


def untransform_video(video_tensor: torch.Tensor) -> np.ndarray:
    """Invert the transformations applied by `transform_video`.

    Args:
        video_tensor: 4D or 5D tensor with shape ([B], C, T, H, W) and values in [0, 1].

    Returns:
        4D or 5D array of video frames with shape ([B], T, H, W, C) and values in [0, 255].
    """
    if video_tensor.ndim == 4:
        video_array = video_tensor.permute(1, 2, 3, 0).numpy()
    elif video_tensor.ndim == 5:
        video_array = video_tensor.permute(0, 2, 3, 4, 1).numpy()
    video_array = (video_array * 255.0).astype(np.uint8)  # Convert to [0, 255]
    return video_array


def crop_image(
    image: np.ndarray, centroid: Tuple[int, int], crop_size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """Crop an image around a centroid.

    Args:
        image: Image to crop as array of shape ``(H, W, C)`` or ``(H, W)``.
        centroid: Tuple of ``(x, y)`` coordinates representing the centroid around which to crop.
        crop_size: Size of the crop. If an integer is provided, it will crop a square of that size.

    Returns:
        Cropped image as array of shape ``(H', W', C)`` or ``(H', W')``.
    """
    if isinstance(crop_size, tuple):
        w, h = crop_size
    else:
        w, h = crop_size, crop_size
    x, y = int(centroid[0]), int(centroid[1])

    x_min = max(0, x - w // 2)
    y_min = max(0, y - h // 2)
    x_max = min(image.shape[1], x + w // 2)
    y_max = min(image.shape[0], y + h // 2)

    cropped = image[y_min:y_max, x_min:x_max]
    padded = np.zeros((h, w, *image.shape[2:]), dtype=image.dtype)
    pad_x = max(w // 2 - x, 0)
    pad_y = max(h // 2 - y, 0)
    padded[pad_y : pad_y + cropped.shape[0], pad_x : pad_x + cropped.shape[1]] = cropped
    return padded


def crop_video(
    video_path: str,
    cropped_video_path: str,
    track: np.ndarray,
    crop_size: Union[int, Tuple[int, int]],
    quality: int = 5,
    constrain_track: Optional[bool] = False,
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


def sample_video_clips(
    video_paths: List[str],
    num_samples: int,
    duration: int = 1,
    video_lengths: Optional[List[int]] = None,
) -> Tuple[List[str], List[int], List[int]]:
    """Uniformly sample clips (path, start frame and end frame) from a list of videos.

    Args:
        video_paths: List of video file paths.
        num_samples: Number of clips to sample.
        duration: Ensure start frames are at least this distance from the end of the video.
        video_lengths: Video lengths in frames. If ``None``, lengths are determined from files.

    Returns:
        Tuple of lists containing:
            - video_paths: Sampled video file paths.
            - start_frames: Start frame indices for each sampled clip.
            - end_frames: End frame indices for each sampled clip (non-inclusive).
    """
    if video_lengths is None:
        video_lengths = [len(OpenCVReader(p)) for p in video_paths]

    p = np.array(video_lengths) + 1 - duration
    video_probabilities = p / np.sum(p)

    video_indexes = np.random.choice(
        len(video_paths), size=num_samples, p=video_probabilities
    )
    start_frames = [
        np.random.randint(0, video_lengths[i] - duration + 1) for i in video_indexes
    ]
    end_frames = [start + duration for start in start_frames]
    return [video_paths[i] for i in video_indexes], start_frames, end_frames


def save_model_config(
    training_dir: str,
    model_name: str,
    crop_size: int,
    duration: int,
    temporal_downsample: float = 1.0,
    spatial_downsample: float = 1.0,
    overwrite: bool = False,
) -> None:
    """Saves configuration parameters of a video embedding model.

    Args:
        training_dir: Directory where the model config is saved.
        model_name: Name of the model (e.g., "s3d").
        crop_size: Crop size used during training (before spatial downsampling).
        duration: Clip duration used during training (before temporal downsampling).
        temporal_downsample: Temporal downsampling factor.
        spatial_downsample: Spatial downsampling factor.
        overwrite: If True, overwrites existing configuration file.
    """
    config_path = os.path.join(training_dir, "config.json")

    config = {
        "model_name": model_name,
        "crop_size": crop_size,
        "duration": duration,
        "temporal_downsample": temporal_downsample,
        "spatial_downsample": spatial_downsample,
    }
    if not overwrite and os.path.exists(config_path):
        raise FileExistsError(
            f"Configuration file {config_path} already exists. Set overwrite=True to replace it."
        )
    else:
        print(f"Saving model configuration to {config_path}")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)


def _extract_epoch(path: str) -> int:
    """Extract epoch from checkpoint filename."""
    match = re.search(r"checkpoint_(\d+)\.pth", os.path.basename(path))
    return int(match.group(1)) if match else -1  # -1 for malformed names


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get path to checkpoint with the highest epoch number.

    Args:
        checkpoint_dir: Directory where checkpoints are saved.

    Returns:
        Path to the latest checkpoint file or None if no checkpoints exist.
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    if checkpoints:
        return max(checkpoints, key=_extract_epoch)
    else:
        return None


def load_video_clip(path: str, start: int, end: int) -> np.ndarray:
    """Read video clip from a file using OpenCV.

    Args:
        path: Path to the video file.
        start: Start frame index for the clip.
        end: End frame index for the clip (non-inclusive).

    Returns:
        Video as array of frames.
    """
    reader = OpenCVReader(path)
    clip = reader[start : end]
    return np.stack(clip)


def downsample_video(
    video_array: np.ndarray,
    temporal_downsample: int = 1,
    spatial_downsample: float = 1.0,
) -> np.ndarray:
    """Downsample a video clip temporally and spatially.

    Args:
        video_array: Video as array of frames.
        temporal_downsample: Factor by which to downsample the temporal dimension.
        spatial_downsample: Factor by which to downsample the spatial dimensions.

    Returns:
        Downsampled video array.
    """
    if temporal_downsample > 1:
        video_array = video_array[:: int(temporal_downsample)]

    if spatial_downsample > 1:
        fx = fy = 1.0 / spatial_downsample
        video_array = np.stack(
            [cv2.resize(frame, (0, 0), fx=fx, fy=fy) for frame in video_array]
        )

    return video_array


def center_crop(video_array: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop video around its center (fast method that uses slicing).

    Args:
        video_array: Video as array of frames.
        crop_size: Size of the crop.

    Note:
        If the video is smaller than crop_size, it will not be cropped.

    Returns:
        Center-cropped video.
    """
    h, w = video_array.shape[1:3]
    if h > crop_size:
        video_array = video_array[:, (h - crop_size) // 2 : -(h - crop_size) // 2]
    if w > crop_size:
        video_array = video_array[:, :, (w - crop_size) // 2 : -(w - crop_size) // 2]
    return video_array


class EmbeddingStore:
    """Store for video embeddings and associated metadata."""
    REQUIRED_METADATA = {"video_path", "start_frame", "end_frame"}

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        """
        embeddings: np.ndarray of shape (n_points, embedding_dim)
        metadata: pd.DataFrame of shape (n_points, n_metadata_fields), must contain REQUIRED_COLUMNS
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self._validate()

    def _validate(self):
        """Validate embeddings and metadata consistency. Raises ValueError if checks fail."""
        # Check matching number of datapoints
        n_emb = self.embeddings.shape[0]
        n_meta = self.metadata.shape[0]
        if n_emb != n_meta:
            raise ValueError(
                f"Mismatch in number of datapoints: embeddings has {n_emb}, metadata has {n_meta}"
            )
        # Check required metadata columns
        missing = self.REQUIRED_METADATA - set(self.metadata.columns)
        if missing:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing)}"
            )

    def save(self, path: str):
        """Save embeddings and metadata to an HDF5 file. Overwrites any existing file."""
        self._validate()

        with h5py.File(path, "w") as f:
            f.create_dataset("embeddings", data=self.embeddings)
            meta_grp = f.create_group("metadata")
            for col in self.metadata.columns:
                data = self.metadata[col].values
                if np.issubdtype(data.dtype, np.str_) or data.dtype == object:
                    data = data.astype(object)
                    dt = h5py.string_dtype(encoding='utf-8')
                elif np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
                    dt = data.dtype
                else:
                    raise TypeError(f"Unsupported dtype in column '{col}': {data.dtype}")
                meta_grp.create_dataset(col, data=data, dtype=dt)


    def __len__(self) -> int:
        """Return the number of embeddings stored."""
        return len(self.embeddings)

    def get_clip_info(self, index: int) -> Tuple[str, int, int]:
        """Get video clip info (path, start frame, end frame) at a given index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for embeddings of length {len(self)}")
        row = self.metadata.iloc[index]
        return row["video_path"], row["start_frame"], row["end_frame"]

    @classmethod
    def load(cls, path: str):
        """Load embeddings and metadata from an HDF5 file and return an EmbeddingData instance."""
        with h5py.File(path, "r") as f:
            embeddings = f["embeddings"][:]
            meta_grp = f["metadata"]
            metadata = pd.DataFrame({key: meta_grp[key][()] for key in meta_grp})
        return cls(embeddings, metadata)


class VideoClipStreamer:
    """Stream video clips of specified duration and spacing (with optional cropping)."""
    
    def __init__(
        self, 
        video_path: str, 
        duration: int, 
        spacing: int = 1,
        crop_size: Optional[int] = None,
        track: Optional[np.ndarray] = None
    ):
        """
        Args:
            video_path: Path to the video file.
            duration: Duration of each clip in frames.
            spacing: Number of frames between the start of each clip.
            crop_size: Optional size of the crop to apply to each frame.
            track: Crop centroid for each frame. If None, frames are center-cropped.
        """
        self.duration = duration
        self.spacing = spacing
        self.reader = OpenCVReader(video_path)
        self.frame_buffer = deque(maxlen=duration)
        self.crop_size = crop_size
        
        if track is None:
            height, width = self.reader[0].shape[1:3]
            self.track = np.array([[width // 2, height // 2]] * len(self.reader), dtype=int)
        else:
            self.track = track

        self.start_frames = np.arange(0, len(self.reader) - duration + 1, spacing)
        self.end_frames = self.start_frames + duration

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over the video, yielding clips of specified duration."""
        for current_ix, frame in enumerate(self.reader):
            if self.crop_size is not None:
                cen = self.track[current_ix]
                frame = crop_image(frame, cen, self.crop_size)

            self.frame_buffer.append(frame)
            start_ix = current_ix - self.duration + 1
            if start_ix in self.start_frames:
                yield np.stack(self.frame_buffer)
                
    def __len__(self) -> int:
        """Return the number of clips that can be generated from the video."""
        return (len(self.reader) - self.duration) // self.spacing