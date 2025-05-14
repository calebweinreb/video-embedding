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
    
class VideoAugmentator():
    """Create class for applying consistent augmentations to a video sequence using albumentations."""
    def __init__(
        self,
        duration=30,
        crop_size=256, 
        drift_prob=0.9,
        gaussian_kernel=15, 
        multiplier=10, 
        dof=1.5,   
    ):
        self.duration = duration
        self.crop_size = crop_size
        self.drift_params = (drift_prob, dof, gaussian_kernel, multiplier)
        
        self.transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),                 # Horizontal flip
            A.VerticalFlip(p=0.5),                   # Vertical flip
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.2,0.1), rotate_limit=360, p=0.8, border_mode = cv2.BORDER_REFLECT),
            A.RandomBrightnessContrast(p=0.5),       # Adjust brightness and contrast
            A.GaussianBlur(blur_limit=(1, 3), p=0.3), # Apply Gaussian blur
            A.HueSaturationValue(p=0.5, hue_shift_limit=5),             # Random hue, saturation, value shifts
            A.RandomGamma(p=0.3),                    # Random gamma adjustment
            A.ColorJitter(p=0.3),                    # Color jittering
            A.MotionBlur(blur_limit=7, p=0.2),       # Motion blur
        ])        

    def __call__(self, video_array):

    """
    Apply temporal crop, drift, augmentations, and spatial crop.

    Args:
        video_array (np.ndarray): Video as array of frames.

    Returns:
        np.ndarray: Augmented video.
    """
        video_array = random_temporal_crop(video_array, self.duration)
        video_array = random_drift(video_array, *self.drift_params)
        video_array = apply_albumentations_to_video(video_array, self.transform)
        video_array = center_crop(video_array, self.crop_size)
        return video_array

class BarlowTwins(torch.nn.Module):
    """ 
    Barlow Twins model for self-supervised learning of video representations.
    This model uses a backbone feature extractor and a projector to learn representations from augmented video sequences.

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
        self.bn = torch.nn.BatchNorm1d(projection_dim, affine=False) #batch normalization layer used to normalize features before computing correlation matrix

        
    def forward(self, x1, x2): #two augmented versions of the same input 
        z1, z2 = self.encoder(x1), self.encoder(x2) #passes both inputs through encoder
        bz = z1.shape[0]
        c = self.bn(z1).T @ self.bn(z2) #computes cross-correlation matrix between normalized outputs
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
                    torch.nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    torch.nn.ReLU(inplace=True),
                    )
        self.layer2 = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
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
    Extracts the off-diagonal elements of a square matrix.

    Args:
        x (torch.Tensor): Input tensor of shape (n, n).

    Returns:
        torch.Tensor: Off-diagonal elements of the input tensor, flattened.
    """
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    