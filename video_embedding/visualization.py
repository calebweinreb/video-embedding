import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
from vidio.read import OpenCVReader
from typing import Optional, Dict, Tuple, Union, Iterable
import torch

from .utils import sample_video_clips, untransform_video, crop_image


def play_videos(
    videos: Iterable[np.ndarray],
    rows: int,
    cols: int,
    inches: int = 3,
    titles: Optional[Iterable[str]] = None,
) -> HTML:
    """Play videos (arranged in a grid) in a jupyter notebook.

    Args:
        videos: List of video clips as arrays with shape (frames, height, width, channels).
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        inches: Size of each subplot in inches.
        titles: Optional list of titles for each video.

    Returns:
        HTML5 video player with the specified videos.
    """
    num_videos = len(videos)
    if titles is not None:
        if len(titles) != len(videos):
            raise ValueError("Number of titles must match number of videos.")

    if rows * cols < num_videos:
        raise ValueError("Grid size (rows * cols) is smaller than the number of videos.")

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * inches, rows * inches), constrained_layout=True
    )

    if rows == 1 or cols == 1:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()

    ims = []
    for i, video in enumerate(videos):
        axes[i].imshow(video[0])
        axes[i].axis("off")
        if titles is not None:
            axes[i].set_title(titles[i])
        im = axes[i].imshow(video[0], animated=True)
        ims.append(im)

    for i in range(num_videos, rows * cols):
        axes[i].axis("off")

    plt.close(fig)

    def init():
        for i, video in enumerate(videos):
            ims[i].set_data(video[0])
        return ims

    def animate(frame):
        for i, video in enumerate(videos):
            ims[i].set_data(video[frame])
        return ims

    num_frames = min(video.shape[0] for video in videos)
    anim = FuncAnimation(
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
    samples = sample_video_clips(video_paths, n_examples)
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


def scatter_with_cluster_labels(
    xy: np.ndarray,
    clus: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap_name: str = "jet",
    point_size: int = 5,
    label_fontsize: int = 10
) -> matplotlib.axes.Axes:
    """
    Create scatter plot with cluster-based color and text labels at cluster medians.

    Args:
        xy: Nx2 array of points to plot.
        clus: N-length array of integer cluster labels.
        ax: Axis to plot on. If None, creates a new one.
        cmap_name: Name of the colormap to use.
        point_size: Size of scatter points.
        label_fontsize: Font size for cluster labels.

    Returns:
        matplotlib.axes.Axes: The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=np.min(clus), vmax=np.max(clus))

    # Scatter points
    ax.scatter(*xy.T, c=clus, cmap=cmap, s=point_size)

    # Add labels at cluster medians
    for cluster_id in np.unique(clus):
        cluster_points = xy[clus == cluster_id]
        centroid = np.median(cluster_points, axis=0)
        color = cmap(norm(cluster_id))
        ax.text(
            *centroid,
            str(cluster_id),
            fontsize=label_fontsize,
            ha='center',
            va='center',
            color=color,
            bbox=dict(facecolor='white', edgecolor=color, pad=2, alpha=0.8)
        )

    ax.axis('off')
    return ax
