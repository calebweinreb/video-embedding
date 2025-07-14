import torch
from typing import Tuple, Union, Optional
from torchvision import models
import numpy as np
import os
import json
from vidio.read import OpenCVReader
from .utils import (
    transform_video,
    get_latest_checkpoint,
    downsample_video,
    crop_image,
)


class BarlowTwins(torch.nn.Module):
    """Barlow Twins model for self-supervised learning of video representations.

    References:
        - Paper: https://arxiv.org/abs/2103.03230
        - Code: https://arxiv.org/abs/2104.02057
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        feature_size: int,
        projection_dim: int = 1024,
        hidden_dim: int = 1024,
        lamda: float = 0.001,
    ):
        """
        Args:
            backbone: Feature extractor.
            feature_size: Size of the features output by the backbone.
            projection_dim: Output dimension of the projector MLP.
            hidden_dim: Hidden layer dimension in the projector.
            lamda: Weighting for the off-diagonal loss term.
        """
        super().__init__()
        self.lamda = lamda
        self.backbone = backbone  # feature extractor

        # neural network mapping extracted features into space suitable for BT loss
        self.projector = Projector(feature_size, hidden_dim, projection_dim)

        # combines backbone and projector into one "encoder" model
        self.encoder = torch.nn.Sequential(self.backbone, self.projector)

        self.bn = torch.nn.BatchNorm1d(projection_dim, affine=False)

    def forward(self, x1, x2):  # two augmented versions of the same input
        """Compute Barlow Twins loss for a pair of augmented clips."""

        # pass both inputs through encoder
        z1, z2 = self.encoder(x1), self.encoder(x2)

        # compute cross-correlation matrix between normalized outputs
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss


class Projector(torch.nn.Module):
    """Maps high-dim features from backbone into a space where Barlow Twins loss can be applied."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        """
        Args:
            in_dim: Input dimension.
            hidden_dim: Hidden dimension.
            out_dim: Output dimension.
            super().__init__()
        """
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim, bias=False),
            torch.nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-5),
            torch.nn.ReLU(inplace=True),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
            torch.nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-5),
            torch.nn.ReLU(inplace=True),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        """Forward pass of the projection head."""

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Extract the off-diagonal elements of a square matrix.

    Args:
        x: Input tensor of shape ``(n, n)``.

    Returns:
        Off-diagonal elements of the input tensor, flattened.
    """
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NormalizeInput(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


def get_embedding_model(name: str = "s3d") -> Tuple[torch.nn.Module, int]:
    """Get a pre-trained video embedding model based on the specified name.

    Note: inputs should be [0, 1] normalized RGB tensors of shape (B, C, T, H, W).

    Args:
        name: Name of the model to retrieve. Currently only "s3d" is supported.
        device: Device on which to place the normalization buffers.

    Returns:
        Tuple of:
            - Model: Pre-trained embedding model with normalization prepended.
            - feature_size: Dimensionality of extracted features.
    """
    if name == "s3d":
        # Instantiate S3D
        model = models.video.s3d(weights=models.video.S3D_Weights.DEFAULT)
        model.avgpool = torch.nn.Identity()
        model.classifier = torch.nn.Identity()
        feature_size = 1024

        # Define normalization layer
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        norm = NormalizeInput(mean, std)

        # Combine into one sequential model
        model = torch.nn.Sequential(norm, model)

    else:
        raise ValueError(f"Model {name} is not supported.")

    return model, feature_size


class VideoEmbedder(torch.nn.Module):
    """Class for preprocessing and embedding video clips using a trained backbone.

    The module can be instantiated directly with a backbone and preprocessing parameters or
    constructed from a saved training run that includes a model configuration and checkpoints.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        crop_size: int,
        duration: int,
        temporal_downsample: float = 1.0,
        spatial_downsample: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            backbone: Trained video embedding model.
            crop_size: Crop size to apply to input video clips (prior to spatial downsampling).
            duration: Duration of input video clips in frames (prior to temporal downsampling).
            temporal_downsample: Factor by which to downsample in time.
            spatial_downsample: Factor by which to downsample spatially.
            device: Device on which embeddings should be computed.
        """
        super().__init__()
        self.backbone = backbone.to(device).eval()
        self.crop_size = crop_size
        self.duration = duration
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.device = device

    @classmethod
    def from_training_run(
        cls,
        training_dir: str,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ) -> "VideoEmbedder":
        """Initialize VideoEmbedder from a training run by loading config and checkpoint.

        Args:
            training_dir: Directory with model config and checkpoints.
            checkpoint_path: Specific checkpoint to load. If None, the latest checkpoint is used.
            device: Device on which embeddings should be computed.

        Returns:
            Initialized ``VideoEmbedder``.
        """
        config_file = os.path.join(training_dir, "config.json")
        with open(config_file, "r") as f:
            cfg = json.load(f)

        backbone, feature_size = get_embedding_model(cfg["model_name"])
        if checkpoint_path is None:
            checkpoint_dir = os.path.join(training_dir, "checkpoints")
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        learner = BarlowTwins(backbone, feature_size)
        learner.load_state_dict(checkpoint["learner_state_dict"])
        backbone = learner.backbone

        return cls(
            backbone=backbone,
            crop_size=cfg["crop_size"],
            duration=cfg["duration"],
            temporal_downsample=cfg["temporal_downsample"],
            spatial_downsample=cfg["spatial_downsample"],
            device=device,
        )

    def forward(self, video: np.ndarray, centroids: Optional[np.ndarray] = None) -> np.ndarray:
        """Preprocess and embed a video clip.

        Args:
            video: Video clip as (T, H, W, C) array.
            centroids: Centroids for cropping as (T, 2) array. If None, frames are center-cropped.

        Returns:
            Feature embedding as a 1D array.
        """
        # ensure video has the correct number of frames
        if video.shape[0] != self.duration:
            raise ValueError(
                f"Expected video with {self.duration} frames, got {video.shape[0]} frames."
            )

        # crop video if necessary
        if video.shape[1] > self.crop_size or video.shape[2] > self.crop_size:
            if centroids is None:
                centroids = np.array([[video.shape[2] // 2, video.shape[1] // 2]] * video.shape[0])
            video = np.stack(
                [crop_image(frame, cen, self.crop_size) for frame, cen in zip(video, centroids)]
            )

        # downsample video in time and space
        video = downsample_video(
            video,
            temporal_downsample=self.temporal_downsample,
            spatial_downsample=self.spatial_downsample,
        )

        # normalize and transform video to tensor
        video_tensor = transform_video(video)

        # embed using the backbone
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device).unsqueeze(0)
            features = self.backbone(video_tensor).squeeze(0)
        return features.detach().cpu().numpy()
