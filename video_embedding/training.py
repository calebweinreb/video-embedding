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
    '''Load, apply augmentations to video clips, and return two augmented versions of the same clip.'''
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
    steps_per_epoch: int= 500, 
    checkpoint_dir:str = 'checkpoint_directory', 
    device: str = "cuda", 
) ->None:
    """
    Training loop for the Barlow Twins model.

    Args:
        learner (torch.nn.Module): Learner model returning loss.
        model (torch.nn.Module): Backbone to extract features.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        dataloader (DataLoader): DataLoader for training data.
        start_epoch (int): Starting epoch for training.
        epochs (int): Total number of epochs to train.
        steps_per_epoch (int): Number of steps per epoch.
        checkpoint_dir (str): Directory to save checkpoints.
        loss_log_path (str): Path to save loss logs.
        device (str, optional): Device to use for training. Defaults to "cuda".

    Returns:
        None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

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

        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'learner_state_dict': learner.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': running_loss / steps_per_epoch,
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pth")
        torch.save(save_dict, checkpoint_path)

    scheduler.step(avg_loss)

def load_from_checkpoint(checkpoint_path, model, learner, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    learner.load_state_dict(checkpoint['learner_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return learner, scheduler, optimizer, model, epoch