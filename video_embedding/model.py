import torch
from typing import Tuple, Union, Optional, Literal
from dataclasses import dataclass, asdict
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

@dataclass
class EmbeddingConfig:
    """Configuration for video embedding model."""
    backbone_type: Literal["s3d"] = "s3d"  # currently only supports S3D
    head_type: Literal["linear", "mlp", "none"] = "linear"
    out_dim: Optional[int] = 64  # only used if head is not "none"
    hidden_dim: Optional[int] = 256  # only used if head is "mlp"
    frozen_backbone: bool = True  # whether to freeze the backbone during training


@dataclass
class BarlowTwinsConfig:
    """Configuration for Barlow Twins model."""
    projection_dim: int = 128
    hidden_dim: int = 512
    lamda: float = 0.001


def save_config(
    training_dir: str,
    embedding_config: EmbeddingConfig,
    barlow_twins_config: BarlowTwinsConfig,
    crop_size: int,
    duration: int,
    temporal_downsample: float = 1.0,
    spatial_downsample: float = 1.0,
    overwrite: bool = False,
) -> None:
    """Saves configuration parameters of a video embedding model.

    Args:
        training_dir: Directory where the model config is saved.
        embedding_config: Configuration for the embedding model.
        barlow_twins_config: Configuration for the Barlow Twins model.
        crop_size: Crop size used during training (before spatial downsampling).
        duration: Clip duration used during training (before temporal downsampling).
        temporal_downsample: Temporal downsampling factor.
        spatial_downsample: Spatial downsampling factor.
        overwrite: If True, overwrites existing configuration file.
    """
    config_path = os.path.join(training_dir, "config.json")

    config = {
        "embedding_config": asdict(embedding_config),
        "barlow_twins_config": asdict(barlow_twins_config),
        "crop_size": crop_size,
        "duration": duration,
        "temporal_downsample": temporal_downsample,
        "spatial_downsample": spatial_downsample,
    }
    if not overwrite and os.path.exists(config_path):
        raise FileExistsError(
            f"Configuration file {config_path} already exists. Set overwrite=True to replace it."
        )
    else:
        print(f"Saving model configuration to {config_path}")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)




class BarlowTwins(torch.nn.Module):
    """Barlow Twins model for self-supervised learning of video representations.

    References:
        - Paper: https://arxiv.org/abs/2103.03230
        - Code: https://arxiv.org/abs/2104.02057
    """

    def __init__(self, config: BarlowTwinsConfig, embedding_model: torch.nn.Module):
        """
        Args:
            config: Configuration for Barlow Twins model.
            embedding_model: Embedding model that extracts features from video clips.
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.lamda = config.lamda
        self.projector = Projector(
            embedding_model.feature_size, config.hidden_dim, config.projection_dim
        )
        self.encoder = torch.nn.Sequential(self.embedding_model, self.projector)
        self.bn = torch.nn.BatchNorm1d(config.projection_dim, affine=False)

    def forward(self, x1, x2):  # two augmented versions of the same input
        """Compute Barlow Twins loss for a pair of augmented clips."""
        # compute cross-correlation matrix
        z1, z2 = self.encoder(x1), self.encoder(x2)
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.shape[0])

        # compute Barlow Twins loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss


class Projector(torch.nn.Module):
    """Maps from embedding space to a projection space where Barlow Twins loss can be applied."""

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


def get_embedding_model(config: EmbeddingConfig) -> torch.nn.Module:
    """Build a video embedding model from a config. 
        - Models are contructed from a (pre-trained) backbone and optional linear/MLP head.
        - Inputs to the model should be 5D tensors of shape (B, C, T, H, W).
    """
    # Create backbone
    if config.backbone_type == "s3d":
        # Instantiate pre-trained S3D
        backbone = models.video.s3d(weights=models.video.S3D_Weights.DEFAULT)
        backbone.avgpool = torch.nn.Identity()
        backbone.classifier = torch.nn.Identity()
        feature_size = 1024

        # Prepend normalization layer
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        norm = NormalizeInput(mean, std)
        backbone = torch.nn.Sequential(norm, backbone)
    else:
        raise ValueError(f"Backbone {config.backbone_type} is not supported.")

    # Optionally freeze the backbone
    if config.frozen_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    # Create head
    if config.head_type == "linear":
        head = torch.nn.Linear(feature_size, config.out_dim)
        feature_size = config.out_dim
    elif config.head_type == "mlp":
        head = torch.nn.Sequential(
            torch.nn.Linear(feature_size, config.hidden_dim, bias=False),
            torch.nn.BatchNorm1d(config.hidden_dim), 
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_dim, config.out_dim, bias=False),
            torch.nn.BatchNorm1d(config.out_dim)
        )
        feature_size = config.out_dim
    elif config.head_type == "none":
        head = torch.nn.Identity()
    else:
        raise ValueError(f"Head type {config.head_type} is not supported.")

    # Combine backbone and head
    embedding_model = torch.nn.Sequential(backbone, head)
    embedding_model.feature_size = feature_size
    return embedding_model


class VideoEmbedder(torch.nn.Module):
    """Class for preprocessing and embedding video clips.

    The module can be instantiated directly with a backbone and preprocessing parameters or
    constructed from a saved training run that includes a model configuration and checkpoints.
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        crop_size: int,
        duration: int,
        temporal_downsample: float = 1.0,
        spatial_downsample: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            embedding_model: Trained video embedding model.
            crop_size: Crop size to apply to input video clips (prior to spatial downsampling).
            duration: Duration of input video clips in frames (prior to temporal downsampling).
            temporal_downsample: Factor by which to downsample in time.
            spatial_downsample: Factor by which to downsample spatially.
            device: Device on which embeddings should be computed.
        """
        super().__init__()
        self.embedding_model = embedding_model.to(device).eval()
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

        embedding_cfg = EmbeddingConfig(**cfg["embedding_config"])
        barlow_twins_cfg = BarlowTwinsConfig(**cfg["barlow_twins_config"])

        embedding_model = get_embedding_model(embedding_cfg)
        if checkpoint_path is None:
            checkpoint_dir = os.path.join(training_dir, "checkpoints")
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        learner = BarlowTwins(barlow_twins_cfg, embedding_model)
        learner.load_state_dict(checkpoint["learner_state_dict"])
        embedding_model = learner.embedding_model

        return cls(
            embedding_model=embedding_model,
            crop_size=cfg["crop_size"],
            duration=cfg["duration"],
            temporal_downsample=cfg["temporal_downsample"],
            spatial_downsample=cfg["spatial_downsample"],
            device=device,
        )

    def forward(
        self, video: np.ndarray, centroids: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
                centroids = np.array(
                    [[video.shape[2] // 2, video.shape[1] // 2]] * video.shape[0]
                )
            video = np.stack(
                [
                    crop_image(frame, cen, self.crop_size)
                    for frame, cen in zip(video, centroids)
                ]
            )

        # downsample and transform video
        video = downsample_video(
            video,
            temporal_downsample=self.temporal_downsample,
            spatial_downsample=self.spatial_downsample,
        )
        video_tensor = transform_video(video)

        # embed video
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device).unsqueeze(0)
            features = self.embedding_model(video_tensor).squeeze(0)
        return features.detach().cpu().numpy()


