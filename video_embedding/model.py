import torch
from typing import Tuple
from torchvision import models


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
        lamda: float = 0.001
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

        on_diag = (torch.diagonal(c).add_(-1).pow_(2).sum())
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


def get_embedding_model(name: str = "s3d") -> Tuple[torch.nn.Module, int]:
    """Get a pre-trained video embedding model based on the specified name.

    Args:
        name: Name of the model to retrieve. Currently the only supported model is "s3d".

    Returns:
        Pre-trained video embedding model and the dimension of the extracted features.
    """
    if name == "s3d":
        model = models.video.s3d(weights=models.video.S3D_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        feature_size = 1024
    else:
        raise ValueError(f"Model {name} is not supported.")
    return model, feature_size
