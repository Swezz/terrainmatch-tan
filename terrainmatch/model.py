"""Siamese CNN model for terrain-patch matching.

Architecture

* A shared convolutional encoder maps a single-channel terrain patch
  (``1 × H × W``) to a fixed-length embedding vector.
* The encoder uses three convolutional blocks (Conv → BN → ReLU → MaxPool)
  followed by adaptive average pooling and two fully-connected layers.
* The Siamese network ties the encoder weights between both branches so that
  the same transformation is applied to the anchor and the comparison patch.
* Embeddings are L2-normalised before similarity comparison, which ensures
  cosine similarity equals the dot product of the normalised vectors.

Loss
----
* Contrastive loss (Hadsell et al., 2006) is used during training.
  For positive pairs (label=1) it penalises large embedding distances;
  for negative pairs (label=0) it penalises distances smaller than the
  margin ``m``.

      L = y * d² + (1 − y) * max(0, m − d)²

  where ``d = ‖e_a − e_b‖₂`` is the Euclidean distance between normalised
  embeddings and ``m`` is the margin hyper-parameter.

Note: This is a prototype for learning terrain representations and patch
matching, not an operational navigation component.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder block helpers
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


# ---------------------------------------------------------------------------
# Terrain Encoder
# ---------------------------------------------------------------------------

class TerrainEncoder(nn.Module):
    """Convolutional encoder that maps a terrain patch to an embedding vector.

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for single-band elevation).
    embedding_dim:
        Length of the output feature vector.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),   # → (32, H/2, W/2)
            _conv_block(32, 64),            # → (64, H/4, W/4)
            _conv_block(64, 128),           # → (128, H/8, W/8)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # → (128, 4, 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of patches.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            L2-normalised embedding of shape ``(B, embedding_dim)``.
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Siamese Network
# ---------------------------------------------------------------------------

class SiameseTerrainNet(nn.Module):
    """Siamese network that compares two terrain patches.

    Both branches share the same :class:`TerrainEncoder` (tied weights).
    The forward pass returns the pair of L2-normalised embeddings.
    Use :func:`contrastive_loss` during training and cosine similarity
    (which equals the dot product for unit vectors) during inference.

    Parameters
    ----------
    embedding_dim:
        Length of each embedding vector.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TerrainEncoder(in_channels=1, embedding_dim=embedding_dim)

    def forward(
        self, anchor: torch.Tensor, sample: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both branches.

        Parameters
        ----------
        anchor, sample:
            Tensors of shape ``(B, 1, H, W)``.

        Returns
        -------
        emb_a, emb_b:
            L2-normalised embeddings of shape ``(B, embedding_dim)``.
        """
        emb_a = self.encoder(anchor)
        emb_b = self.encoder(sample)
        return emb_a, emb_b

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single batch of patches."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Contrastive loss (Hadsell et al., 2006) on L2-normalised embeddings.

    Parameters
    ----------
    emb_a, emb_b:
        Normalised embedding tensors of shape ``(B, D)``.
    labels:
        Float tensor of shape ``(B,)`` with ``1`` for positive pairs and
        ``0`` for negative pairs.
    margin:
        Margin for negative pairs.  Should be chosen relative to the
        L2-distance scale (≈ ``sqrt(2)`` for unit-sphere embeddings).

    Returns
    -------
    torch.Tensor
        Scalar mean loss over the batch.
    """
    d = torch.norm(emb_a - emb_b, p=2, dim=1)
    pos = labels * d.pow(2)
    neg = (1 - labels) * F.relu(margin - d).pow(2)
    return (pos + neg).mean()
