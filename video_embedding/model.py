"""Necessities for model implementation."""
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter
import random
from albumentations.pytorch import ToTensorV2
from albumentations import ReplayCompose
from vidio.read import OpenCVReader
from torch.utils.data import DataLoader
import os
import tqdm
from typing import Optional, Dict, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau 
from torchvision import models
    
    
class BarlowTwins(torch.nn.Module):
    """ 
    Barlow Twins model for self-supervised learning of video representations. This model uses a backbone feature extractor and a projector to learn representations from augmented video sequences.

    Args:
        backbone (torch.nn.Module): Backbone feature extractor.
        feature_size (int): Size of the features extracted by the backbone.
        projection_dim (int): Dimension of the projected features.
        hidden_dim (int): Dimension of the hidden layer in the projector.
        lamda (float): Regularization parameter for off-diagonal loss.

    Returns:
        torch.nn.Module: Barlow Twins model.
    
    Barlow Twins
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://arxiv.org/abs/2103.03230
    """
    def __init__(self, backbone, feature_size, projection_dim=1024, hidden_dim=1024, lamda=0.001):
        super().__init__()
        self.lamda = lamda
        self.backbone = backbone #feature extractor
        self.projector = Projector(feature_size, hidden_dim, projection_dim) #neural network mapping extracted features into space suitable for BT loss
        self.encoder = torch.nn.Sequential(self.backbone, self.projector) #combines backbone and projector into one model 
        self.ln = torch.nn.LayerNorm (projection_dim, elementwise_affine=False)

    def forward(self, x1, x2): #two augmented versions of the same input 
        z1, z2 = self.encoder(x1), self.encoder(x2) #passes both inputs through encoder
        bz = z1.shape[0]
        c = self.ln(z1).T @ self.ln(z2) #computes cross-correlation matrix between normalized outputs
        c.div_(bz)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() #applies BT loss; forces diagonal to be 1 causing same features to match
        off_diag = off_diagonal(c).pow_(2).sum() #penalizes non-diagonal elements thereby reducing redundancy between features
        loss = on_diag + self.lamda * off_diag
        return loss


class Projector(torch.nn.Module):
    """ 
    Small feedforward neural model mapping high-dimensional features from the backbone into a space where Barlow Twins loss can be applied.

    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        out_dim (int): Output dimension.    

    Returns:
        torch.nn.Module: Projector model.
    """
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
     
        self.layer1 = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim, bias=False),
                    torch.nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    torch.nn.ReLU(inplace=True),
                    )
        self.layer2 = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                    torch.nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    torch.nn.ReLU(inplace=True),
                    )
        self.layer3 = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, out_dim, bias=False),
                    )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    

def off_diagonal(x):
    """
    Extract the off-diagonal elements of a square matrix.

    Args:
        x (torch.Tensor): Input tensor of shape (n, n).

    Returns:
        torch.Tensor: Off-diagonal elements of the input tensor, flattened.
    """
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

def setup_optim_scheduler(
    learner: torch.nn.Module,
    lr: float = 1e-4,
    scheduler_params: Optional[Dict] = None
) -> Tuple[Optimizer, _LRScheduler]:
    """
    Create an Adam optimizer on learner.parameters() and
    a ReduceLROnPlateau scheduler.
    """
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    sched_kwargs = dict(mode="min", threshold=0.1)
    if scheduler_params:
        sched_kwargs.update(scheduler_params)
    scheduler = ReduceLROnPlateau(opt, **sched_kwargs)
    return opt, scheduler

def train (
    learner:torch.nn.Module,
    model:torch.nn.Module, 
    optimizer:torch.optim.Optimizer, 
    scheduler:torch.optim.lr_scheduler._LRScheduler, 
    dataloader: DataLoader, 
    start_epoch: int, 
    epochs:int, 
    steps_per_epoch: int, 
    checkpoint_dir:str, 
    loss_log_path: str, 
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

    opt = optimizer
    sched = scheduler

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


def get_model(name: str = "s3d"):
    """
    Get a pre-trained video embedding model based on the specified name.

    Args:
        name (str): Name of the model to retrieve. Currently the only supported model is "s3d".

    Returns:
        torch.nn.Module: Pre-trained video embedding model.
        int: Dimension of the features extracted by the model.
    """
    if name == "s3d":
        model = models.video.s3d(weights=models.video.S3D_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        feature_size = 1024
    else:
        raise ValueError(f"Model {name} is not supported.")
    return model, feature_size