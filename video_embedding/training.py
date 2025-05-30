"""Training script for Barlow Twins model on video data."""
import os
import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Optional, Dict, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from albumentations.pytorch import ToTensorV2
from albumentations import ReplayCompose
from .model import BarlowTwins, Projector, off_diagonal
from vidio.read import OpenCVReader
from .utils import transform_video, untransform_video

class VidioDataset(Dataset):
    '''Class for loading video clips and applying augmentations.'''
    def __init__(
        self, video_paths, augmentator, clip_size, 
        temporal_downsample=1, spatial_downsample=1
    ):
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.video_paths = video_paths
        self.augmentator = augmentator
        self.clip_size = clip_size
        lengths = [len(OpenCVReader(p)) for p in video_paths]
        self.video_ixs = np.hstack([torch.ones(n-clip_size)*i for i,n in enumerate(lengths)]).astype(int)
        self.frame_ixs = np.hstack([torch.arange(n-clip_size) for i,n in enumerate(lengths)]).astype(int)
        
    def __len__(self):
        return len(self.video_ixs)

    def __getitem__(self, idx):
        video_ix = self.video_ixs[idx]
        frame_ix = self.frame_ixs[idx]
        reader = OpenCVReader(self.video_paths[video_ix])
        frames = reader[frame_ix : frame_ix + self.clip_size][::self.temporal_downsample]

        if self.spatial_downsample > 1:
            fx = fy = 1./self.spatial_downsample
            frames = [cv2.resize(frame, (0,0), fx=fx, fy=fy) for frame in frames]

        frames = np.stack(frames)
        x_one = transform_video(self.augmentator(frames))
        x_two = transform_video(self.augmentator(frames))
        return x_one, x_two

def train (
    learner:torch.nn.Module,
    model:torch.nn.Module, 
    optimizer:torch.optim.Optimizer, 
    scheduler:torch.optim.lr_scheduler._LRScheduler, 
    dataloader: DataLoader, 
    start_epoch:int = 0,
    epochs:int = 1500, 
    steps_per_epoch: int = 500, 
    checkpoint_dir:str = 'checkpoint_directory', 
    loss_log_path: str = 'loss_log.txt',
    device: str = "cuda", 
) ->None:
    """
    Trains a video embedding model using a Barlow Twins approach.

    Args:
        learner (torch.nn.Module): Learner model returning loss.
        model (torch.nn.Module): Backbone to extract features.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        dataloader (DataLoader): DataLoader for training data.
        start_epoch (int): Starting epoch for training.
        epochs (int): Total number of epochs to train plus starting epoch.
        steps_per_epoch (int): Number of steps per epoch.
        checkpoint_dir (str): Directory to save checkpoints.
        device (str, optional): Device to use for training. Defaults to "cuda".
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_log_path = os.path.join(checkpoint_dir, "log_loss.txt")
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as f:
            f.write("epoch\tloss\n")

    for epoch in range(start_epoch, epochs):
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
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch}\t{avg_loss}\n")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'learner_state_dict': learner.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'scheduler_state_dict': scheduler.state_dict()
        },f'{checkpoint_dir}/checkpoint_{epoch+1}.pth')
    
        scheduler.step(avg_loss)


def load_from_checkpoint(checkpoint_path, model, learner, optimizer, scheduler):
    """
    Load model, learner, optimizer, and scheduler states from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load state into.
        learner (torch.nn.Module): Learner to load state into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to load state into.

    Returns:
        Tuple containing
            - learner (torch.nn.Module): Learner with loaded state.
            - scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler with loaded state.
            - optimizer (torch.optim.Optimizer): Optimizer with loaded state.
            - model (torch.nn.Module): Model with loaded state.
            - epoch (int): Epoch number from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    learner.load_state_dict(checkpoint['learner_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return learner, scheduler, optimizer, model, epoch