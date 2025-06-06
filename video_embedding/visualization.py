import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from vidio.read import OpenCVReader
from typing import Optional, Dict, Tuple, Union, Iterable
import matplotlib
import torch

from .utils import sample_timepoints, untransform_video, crop_image


def play_videos(
    videos: Iterable[np.ndarray], rows: int, cols: int, inches: int = 3
) -> HTML:
    """Play videos (arranged in a grid) in a jupyter notebook.

    Args:
        videos: List of vide clips as arrays with shape ``(frames, height, width, channels)``.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        inches: Size of each subplot in inches.

    Returns:
        HTML5 video player with the specified videos.
    """
    num_videos = len(videos)

    if rows * cols < num_videos:
        raise ValueError(
            "Grid size (rows * cols) is smaller than the number of videos."
        )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * inches, rows * inches))

    if rows == 1:
        axes = axes.flatten() if num_videos > 1 else [axes]
    elif cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    ims = []
    for i, video in enumerate(videos):
        axes[i].imshow(video[0, :, :, :])
        axes[i].axis("off")
        im = axes[i].imshow(video[0, :, :, :], animated=True)
        ims.append(im)

    for i in range(num_videos, rows * cols):
        axes[i].axis("off")

    plt.close(fig)

    def init():
        for i, video in enumerate(videos):
            ims[i].set_data(video[0, :, :, :])
        return ims

    def animate(frame):
        for i, video in enumerate(videos):
            ims[i].set_data(video[frame, :, :, :])
        return ims

    num_frames = min(video.shape[0] for video in videos)
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=num_frames, interval=50, blit=True
    )

    return HTML(anim.to_html5_video())

def inspect_crop_sizes(
    tracks: Dict[str, np.ndarray],
    inner_crop_size: int,
    outer_crop_size: int,
    n_examples: Optional[int] = 4,
) -> matplotlib.figure.Figure:
    """Visualize crop sizes for a random sample of frames.

    Args:
        tracks: Dict from video paths to tracks containing the animal's centroid at each frame.
        inner_crop_size: Size of the inner crop.
        outer_crop_size: Size of the outer crop.
        n_examples: Number of examples to visualize.

    Returns:
        Figure containing the visualizations.
    """
    video_paths = list(tracks.keys())
    samples = sample_timepoints(video_paths, n_examples)
    box = np.array([[-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1]]).T / 2

    fig, axes = plt.subplots(2, n_examples, sharey="row")
    for i, (video_path, frame_ix) in enumerate(samples):
        frame = OpenCVReader(video_path)[frame_ix]
        cen = tracks[video_path][frame_ix]

        # full frame
        axes[0, i].imshow(frame)
        axes[0, i].plot(*(box * outer_crop_size + cen).T, c="b", lw=2)
        axes[0, i].plot(*(box * inner_crop_size + cen).T, c="b", lw=2, ls="--")

        # cropped frame
        cropped_frame = crop_image(frame, cen, outer_crop_size)
        axes[1, i].imshow(cropped_frame)
        axes[1, i].plot(
            *(box * inner_crop_size + outer_crop_size / 2).T, c="b", lw=2, ls="--"
        )

    for ax in axes.flat:
        ax.set_facecolor("black")

    return fig


def inspect_dataloader(
    dataloader: Union[torch.utils.data.DataLoader, Iterable],
    num_samples: int = 4,
    inches: int = 3,
) -> HTML:
    """Visualize a batch of augmented video clip pairs from a dataloader.

    Args:
        dataloader: Dataloader or iterable yielding batched pairs of augmented video clips.
        num_samples: Number of samples to visualize.
        inches: Size of each subplot in inches.

    Returns:
        HTML5 video player displaying the video clips.
    """
    x_one, x_two = next(iter(dataloader))
    if x_one.shape[0] < num_samples:
        raise ValueError(
            f"Batch size {x_one.shape[0]} is less than requested number of samples {num_samples}."
        )
    x_one = untransform_video(x_one)[:num_samples]
    x_two = untransform_video(x_two)[:num_samples]
    return play_videos(np.concatenate([x_one, x_two]), 2, num_samples, inches)
