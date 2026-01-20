"""Model architectures."""

from .fusion_model import DualEmbeddingFusionModel
from .heads import ClassificationHead, AttentionPooling

__all__ = [
    "DualEmbeddingFusionModel",
    "ClassificationHead",
    "AttentionPooling",
]
