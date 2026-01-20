"""
Loss functions for handling class imbalance.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focuses training on hard examples by down-weighting easy examples.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Class weights, shape (num_classes,)
            gamma: Focusing parameter (gamma=0 is equivalent to CE)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes), raw logits
            targets: (batch,), class indices

        Returns:
            loss: scalar or (batch,) depending on reduction
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function for loss functions.

    Args:
        loss_type: 'weighted_ce' or 'focal'
        class_weights: Class weights for imbalance handling
        focal_gamma: Gamma parameter for focal loss

    Returns:
        Loss function module
    """
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
