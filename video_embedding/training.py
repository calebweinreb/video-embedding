import os
import tqdm
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .model import BarlowTwins, Projector, off_diagonal
from vidio.read import OpenCVReader
from .utils import transform_video, untransform_video
from .augmentation import VideoClipAugmentator


class VideoClipDataset(Dataset):
    """Class for loading video clips and applying augmentations."""

    def __init__(
        self,
        video_paths: list[str],
        augmentator: VideoClipAugmentator,
        duration: int,
        temporal_downsample: float = 1.0,
        spatial_downsample: float = 1.0,
    ):
        """
        Args:
            video_paths: List of paths to video files.
            augmentator: VideoClipAugmentator instance for applying augmentations.
            duration: Duration of loaded video clips prior to augmentation.
            temporal_downsample: Factor by which to reduce time dimension (after augmentation).
            spatial_downsample: Factor by which to reduce spatial dimensions (after augmentation).
        """
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.video_paths = video_paths
        self.augmentator = augmentator
        self.duration = duration
        lengths = [len(OpenCVReader(p)) for p in video_paths]
        self.video_ixs = np.hstack(
            [torch.ones(n - duration) * i for i, n in enumerate(lengths)]
        ).astype(int)
        self.frame_ixs = np.hstack(
            [torch.arange(n - duration) for i, n in enumerate(lengths)]
        ).astype(int)

    def __len__(self):
        return len(self.video_ixs)

    def __getitem__(self, idx):
        video_ix = self.video_ixs[idx]
        frame_ix = self.frame_ixs[idx]
        reader = OpenCVReader(self.video_paths[video_ix])
        frames = reader[frame_ix : frame_ix + self.duration][
            :: self.temporal_downsample
        ]

        if self.spatial_downsample > 1:
            fx = fy = 1.0 / self.spatial_downsample
            frames = [cv2.resize(frame, (0, 0), fx=fx, fy=fy) for frame in frames]

        frames = np.stack(frames)
        x_one = transform_video(self.augmentator(frames))
        x_two = transform_video(self.augmentator(frames))
        return x_one, x_two


def train(
    learner,
    optimizer,
    scheduler,
    dataloader,
    num_epochs,
    steps_per_epoch,
    checkpoint_dir,
    device,
) -> None:
    """Trains a video embedding model using a Barlow Twins approach.

    Args:
        learner: Learner model returning loss.
        optimizer: Optimizer for the model.
        scheduler: Learning rate scheduler.
        dataloader: DataLoader for training data.
        num_epochs: Total number of epochs to train plus starting epoch.
        steps_per_epoch: Number of steps per epoch.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to use for training.
    """
    print(f"Saving checkpoints to {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_log = os.path.join(checkpoint_dir, "loss_log.csv")
    print(f"Saving losses to {loss_log}")
    if not os.path.exists(loss_log):
        with open(loss_log, "w") as f:
            f.write("epoch,loss\n")

    previous_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    if previous_checkpoints:
        latest_checkpoint = max(
            previous_checkpoints, key=lambda x: int(re.search(r"(\d+)", x).group(0))
        )
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        learner.load_state_dict(checkpoint["learner_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from checkpoint {latest_checkpoint}")

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        loader = iter(dataloader)

        with tqdm.trange(steps_per_epoch, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")

            for i in tepoch:
                x_one, x_two = next(loader)
                x_one = x_one.to(device)
                x_two = x_two.to(device)

                loss = learner(x_one, x_two)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (i + 1))

        avg_loss = running_loss / steps_per_epoch
        scheduler.step(avg_loss)

        with open(loss_log, "a") as f:
            f.write(f"{epoch},{avg_loss}\n")

        torch.save(
            {
                "epoch": epoch,
                "learner_state_dict": learner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
        )
