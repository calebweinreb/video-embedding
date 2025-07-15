import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Union, Optional, Iterator
from vidio.read import OpenCVReader
import imageio
import tqdm
import h5py
import os
import cv2
import json
import re
import glob
from collections import deque


def transform_video(video_array: np.ndarray) -> torch.Tensor:
    """Permute axes (… T H W C) → (… C T H W), 0-1 normalize, and return as contiguous tensor."""
    video_array = video_array.astype(np.float32) / 255.0
    video_array = np.moveaxis(video_array, -1, -4)
    return torch.tensor(video_array)


def untransform_video(video_tensor: torch.Tensor) -> np.ndarray:
    """Invert the transformations applied by `transform_video`."""
    video_array = np.moveaxis(video_tensor.numpy(), -4, -1)
    return (video_array * 255.0).astype(np.uint8)


def crop_image(
    image: np.ndarray,
    centroid: Tuple[int, int],
    crop_size: Union[int, Tuple[int, int]],
    border_mode: int = cv2.BORDER_REFLECT,
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> np.ndarray:
    """Crop an image around a centroid, using OpenCV border modes for padding.

    Args:
        image:       Input image, shape (H, W) or (H, W, C).
        centroid:    (x, y) coordinates of the crop center.
        crop_size:   Either an int (square) or (width, height).
        border_mode: OpenCV border mode (e.g. cv2.BORDER_REFLECT).
        border_value: Value for BORDER_CONSTANT; scalar or tuple for multi-channel.

    Returns:
        Cropped patch of size exactly (height, width, ...) or (height, width).
    """
    # determine crop width and height
    if isinstance(crop_size, tuple):
        w, h = crop_size
    else:
        w = h = crop_size

    x, y = int(centroid[0]), int(centroid[1])
    H, W = image.shape[:2]
    half_w, half_h = w // 2, h // 2

    # compute how much padding is needed on each side
    top = max(half_h - y, 0)
    bottom = max((y + half_h) - (H - 1), 0)
    left = max(half_w - x, 0)
    right = max((x + half_w) - (W - 1), 0)

    # pad if necessary
    if any((top, bottom, left, right)):
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, borderType=border_mode, value=border_value
        )
        # shift centroid to account for padding
        x += left
        y += top

    # now crop exactly w×h around (x, y)
    x0, y0 = x - half_w, y - half_h
    x1, y1 = x0 + w, y0 + h
    return image[y0:y1, x0:x1]


def crop_video(
    video_path: str,
    cropped_video_path: str,
    track: np.ndarray,
    crop_size: Union[int, Tuple[int, int]],
    quality: int = 5,
    constrain_track: Optional[bool] = False,
    border_mode: int = cv2.BORDER_REFLECT,
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> None:
    """Crop a video around a time-varying centroid.

    Args:
        video_path: Path to the input video.
        cropped_video_path: Path to save the cropped video.
        track: Array of shape ``(frames, 2)`` containing the crop centroid ``(x, y)`` for each frame.
        crop_size: Size of the crop. If an integer is provided, it crops a square of that size.
        quality: Quality of the output video passed to ``imageio.get_writer``.
        constrain_track: If ``True``, ensures the cropped area does not exceed the video boundaries.
        border_mode: OpenCV border mode for padding (e.g., cv2.BORDER_REFLECT).
        border_value: Value for BORDER_CONSTANT; scalar or tuple for multi-channel images.
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
            cropped_frame = crop_image(frame, cen, crop_size, border_mode, border_value)
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
    clip = reader[start:end]
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
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

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
                    dt = h5py.string_dtype(encoding="utf-8")
                elif np.issubdtype(data.dtype, np.number) or np.issubdtype(
                    data.dtype, np.bool_
                ):
                    dt = data.dtype
                else:
                    raise TypeError(
                        f"Unsupported dtype in column '{col}': {data.dtype}"
                    )
                meta_grp.create_dataset(col, data=data, dtype=dt)

    def __len__(self) -> int:
        """Return the number of embeddings stored."""
        return len(self.embeddings)

    def get_clip_info(self, index: int) -> Tuple[str, int, int]:
        """Get video clip info (path, start frame, end frame) at a given index."""
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of bounds for embeddings of length {len(self)}"
            )
        row = self.metadata.iloc[index]
        return row["video_path"], row["start_frame"], row["end_frame"]

    @classmethod
    def load(cls, path: str):
        """Load embeddings and metadata from an HDF5 file and return an EmbeddingData instance."""
        with h5py.File(path, "r") as f:
            embeddings = f["embeddings"][()]
            meta_grp = f["metadata"]
            mdict = {}
            for key in meta_grp:
                dset = meta_grp[key]
                if dset.dtype.kind in {"S", "O"}:
                    mdict[key] = dset.asstr()[()]
                else:
                    mdict[key] = dset[()]
            metadata = pd.DataFrame(mdict)
        return cls(embeddings, metadata)


class VideoClipStreamer:
    """Stream video clips of specified duration and spacing (with optional cropping)."""

    def __init__(
        self,
        video_path: str,
        duration: int,
        spacing: int = 1,
        crop_size: Optional[int] = None,
        track: Optional[np.ndarray] = None,
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
            self.track = np.array(
                [[width // 2, height // 2]] * len(self.reader), dtype=int
            )
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
        return len(self.start_frames)

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