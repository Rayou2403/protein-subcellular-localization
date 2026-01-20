"""
Model heads and pooling layers.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Learnable attention-based pooling.

    Computes attention weights over sequence and returns weighted sum.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            embeddings: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len), 1 for valid, 0 for padding

        Returns:
            pooled: (batch, hidden_dim)
        """
        # Compute attention scores
        attn_scores = self.attention(embeddings).squeeze(-1)  # (batch, seq_len)

        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        # Softmax to get weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            embeddings,  # (batch, seq_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)

        return pooled


class ClassificationHead(nn.Module):
    """
    MLP classification head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch, num_classes)
        """
        return self.fc(x)
