"""
Dual-Embedding Fusion Model for protein localization.

Architecture for pooled embeddings (fixed-size vectors):
1. Separate projections for ESM-C and ProstT5 embeddings
2. Gated fusion mechanism
3. Classification head
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ClassificationHead, AttentionPooling


class GatedFusion(nn.Module):
    """
    Gated fusion for combining two embedding modalities (pooled vectors).
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Cross-modal interaction
        self.cross_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cross_b = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply gated fusion.

        Args:
            emb_a: (batch, hidden_dim)
            emb_b: (batch, hidden_dim)

        Returns:
            fused_a: Enhanced embedding A
            fused_b: Enhanced embedding B
            gate_weights: Gating weights for interpretability
        """
        # Cross-modal interaction
        combined = torch.cat([emb_a, emb_b], dim=-1)
        emb_a_enhanced = emb_a + self.cross_a(combined)
        emb_b_enhanced = emb_b + self.cross_b(combined)

        # Gated fusion
        gate_weights = self.gate(torch.cat([emb_a_enhanced, emb_b_enhanced], dim=-1))

        fused_a = gate_weights * emb_a_enhanced + (1 - gate_weights) * emb_b_enhanced
        fused_b = (1 - gate_weights) * emb_a_enhanced + gate_weights * emb_b_enhanced

        return fused_a, fused_b, gate_weights


class DualEmbeddingFusionModel(nn.Module):
    """
    Complete model for dual-embedding protein localization.

    Architecture for pooled (fixed-size) embeddings:
    - Separate projections for each modality
    - Gated fusion with interpretable weights
    - Classification head
    """

    def __init__(
        self,
        esmc_dim: int = 1280,
        prostt5_dim: int = 1024,
        hidden_dim: int = 512,
        num_attention_heads: int = 8,  # kept for config compatibility
        num_fusion_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 6,
        pooling_type: str = "attention",  # kept for config compatibility
        use_gated_fusion: bool = True,
    ):
        super().__init__()

        self.esmc_dim = esmc_dim
        self.prostt5_dim = prostt5_dim
        self.hidden_dim = hidden_dim
        self.use_gated_fusion = use_gated_fusion

        # Embedding projections
        self.esmc_projection = nn.Sequential(
            nn.Linear(esmc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prostt5_projection = nn.Sequential(
            nn.Linear(prostt5_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated fusion layers
        if use_gated_fusion:
            self.fusion_layers = nn.ModuleList([
                GatedFusion(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_fusion_layers)
            ])
        else:
            self.fusion_layers = nn.ModuleList()

        # Classification head
        pooled_dim = hidden_dim * 2  # Concatenate both modalities

        self.classifier = ClassificationHead(
            input_dim=pooled_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        esmc_embeddings: torch.Tensor,
        prostt5_embeddings: torch.Tensor,
        **kwargs,  # Accept and ignore mask arguments for compatibility
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            esmc_embeddings: (batch, esmc_dim) - pooled embeddings
            prostt5_embeddings: (batch, prostt5_dim) - pooled embeddings

        Returns:
            Dictionary with:
                - logits: (batch, num_classes)
                - gate_weights: (batch, hidden_dim) if using gated fusion
        """
        # Project embeddings
        esmc_proj = self.esmc_projection(esmc_embeddings)
        prostt5_proj = self.prostt5_projection(prostt5_embeddings)

        # Apply fusion layers
        gate_weights_all = []
        if self.use_gated_fusion:
            for fusion_layer in self.fusion_layers:
                esmc_proj, prostt5_proj, gate_weights = fusion_layer(
                    esmc_proj,
                    prostt5_proj,
                )
                gate_weights_all.append(gate_weights)

        # Concatenate modalities
        combined = torch.cat([esmc_proj, prostt5_proj], dim=-1)

        # Classification
        logits = self.classifier(combined)

        output = {"logits": logits}
        if gate_weights_all:
            output["gate_weights"] = torch.stack(gate_weights_all, dim=1)

        return output
