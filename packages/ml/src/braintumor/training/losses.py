"""Loss functions for BraTS segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss.

    Standard for BraTS-style segmentation. Dice handles class imbalance,
    CE provides stable gradients early in training.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.smooth = smooth

    def dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-class soft Dice."""
        probs = torch.softmax(logits, dim=1)
        n_classes = logits.shape[1]

        # target: (B, H, W) or (B, D, H, W)
        # one_hot: (B, H, W, C) or (B, D, H, W, C)
        target_one_hot = F.one_hot(target, num_classes=n_classes).float()

        # Move channel dim to index 1: (B, C, H, W) or (B, C, D, H, W)
        # The last dimension is the class dim.
        dims = list(range(target_one_hot.ndim))
        channel_dim = dims.pop()
        dims.insert(1, channel_dim)
        target_one_hot = target_one_hot.permute(*dims)

        # Reduction dimensions: batch and all spatial dimensions
        # Example 2D: (0, 2, 3)
        # Example 3D: (0, 2, 3, 4)
        reduction_dims = [0] + list(range(2, target_one_hot.ndim))

        intersection = (probs * target_one_hot).sum(dim=reduction_dims)
        cardinality = (probs + target_one_hot).sum(dim=reduction_dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if not self.include_background:
            dice = dice[1:]

        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        dl = self.dice_loss(logits, target)
        return self.dice_weight * dl + self.ce_weight * ce
