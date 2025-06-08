import os
import tqdm
import cv2
import numpy as np
import torch
import json
import glob
import re
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, List, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .model import BarlowTwins, Projector, off_diagonal
from vidio.read import OpenCVReader
from .utils import (
    transform_video,
    untransform_video,
    get_latest_checkpoint,
    load_video_clip,
    downsample_video,
)
from .augmentation import VideoClipAugmentator


class VideoClipDataset(Dataset):
    """Class for loading video clips and applying augmentations."""

    def __init__(
        self,
        video_paths: list[str],
        augmentator: VideoClipAugmentator,
        duration: int,
        temporal_downsample: int = 1,
        spatial_downsample: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            video_paths: List of paths to video files.
            augmentator: VideoClipAugmentator instance for applying augmentations.
            duration: Duration of loaded video clips prior to augmentation.
            temporal_downsample: Factor by which to reduce time dimension (prior to augmentation).
            spatial_downsample: Factor by which to reduce space dimensions (prior to augmentation).
            device: Device on which the dataset will be used.
        """
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.video_paths = video_paths
        self.augmentator = augmentator
        self.duration = duration
        self.device = device

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
        frames = load_video_clip(self.video_paths[video_ix], frame_ix, self.duration)
        frames = downsample_video(
            frames, self.temporal_downsample, self.spatial_downsample
        )
        x_one = transform_video(self.augmentator(frames)[None], device=self.device)
        x_two = transform_video(self.augmentator(frames)[None], device=self.device)
        return x_one.squeeze(0), x_two.squeeze(0)


def train(
    training_dir: str,
    learner: BarlowTwins,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    dataloader: DataLoader,
    num_epochs: int,
    steps_per_epoch: int,
    device: str = "cuda",
) -> None:
    """Trains a video embedding model using a Barlow Twins approach.

    Args:
        training_dir: Directory where checkpoints and loss log are saved.
        learner: Learner model returning loss.
        optimizer: Optimizer for the model.
        scheduler: Learning rate scheduler.
        dataloader: DataLoader for training data.
        num_epochs: Total number of epochs to train.
        steps_per_epoch: Number of steps per epoch.
        device: Device to use for training.
    """
    checkpoint_dir = os.path.join(training_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to {checkpoint_dir}")

    loss_log = os.path.join(training_dir, "loss_log.csv")
    print(f"Saving losses to {loss_log}")
    if not os.path.exists(loss_log):
        with open(loss_log, "w") as f:
            f.write("epoch,loss\n")

    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        learner.load_state_dict(checkpoint["learner_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from checkpoint {latest_checkpoint}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        loader = iter(dataloader)

        with tqdm.trange(steps_per_epoch, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

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
            os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"),
        )
