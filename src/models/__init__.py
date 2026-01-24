"""Model architectures."""

from .fusion_model import DualEmbeddingFusionModel, TransformerFusionModel
from .heads import ClassificationHead, AttentionPooling

__all__ = [
    "DualEmbeddingFusionModel",
    "TransformerFusionModel",
    "ClassificationHead",
    "AttentionPooling",
]
