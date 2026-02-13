"""
Unit tests for model forward pass and data pipeline.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DualEmbeddingFusionModel
from src.data.collate import collate_variable_length


def test_model_forward_pass():
    """Test model forward pass with pooled embeddings."""
    batch_size = 4
    esmc_dim = 960
    prostt5_dim = 1024
    hidden_dim = 256
    num_classes = 6

    model = DualEmbeddingFusionModel(
        esmc_dim=esmc_dim,
        prostt5_dim=prostt5_dim,
        hidden_dim=hidden_dim,
        num_attention_heads=8,
        num_fusion_layers=2,
        dropout=0.3,
        num_classes=num_classes,
        pooling_type="attention",
        use_gated_fusion=True,
    )

    # Dummy input
    esmc_emb = torch.randn(batch_size, esmc_dim)
    prostt5_emb = torch.randn(batch_size, prostt5_dim)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(esmc_emb, prostt5_emb)

    # Check shapes
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, num_classes)
    assert "gate_weights" in outputs

    print("Model forward pass test passed!")


def test_pooled_collate():
    """Test collate function with pooled embeddings."""
    # Create dummy batch
    batch = [
        {
            "esmc_embedding": torch.randn(960),
            "prostt5_embedding": torch.randn(1024),
            "label": torch.tensor(0),
            "protein_id": "P001",
        },
        {
            "esmc_embedding": torch.randn(960),
            "prostt5_embedding": torch.randn(1024),
            "label": torch.tensor(1),
            "protein_id": "P002",
        },
        {
            "esmc_embedding": torch.randn(960),
            "prostt5_embedding": torch.randn(1024),
            "label": torch.tensor(2),
            "protein_id": "P003",
        },
    ]

    # Collate
    collated = collate_variable_length(batch)

    # Check shapes
    assert collated["esmc_embeddings"].shape == (3, 960)
    assert collated["prostt5_embeddings"].shape == (3, 1024)
    assert collated["labels"].shape == (3,)
    assert len(collated["protein_ids"]) == 3

    print("Pooled collate test passed!")


def test_model_with_collated_batch():
    """Test model with collated pooled batch."""
    batch = [
        {
            "esmc_embedding": torch.randn(960),
            "prostt5_embedding": torch.randn(1024),
            "label": torch.tensor(0),
            "protein_id": "P001",
        },
        {
            "esmc_embedding": torch.randn(960),
            "prostt5_embedding": torch.randn(1024),
            "label": torch.tensor(1),
            "protein_id": "P002",
        },
    ]

    collated = collate_variable_length(batch)

    model = DualEmbeddingFusionModel(
        esmc_dim=960,
        prostt5_dim=1024,
        hidden_dim=256,
        num_attention_heads=8,
        num_fusion_layers=2,
        dropout=0.3,
        num_classes=6,
        pooling_type="attention",
        use_gated_fusion=True,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(
            collated["esmc_embeddings"],
            collated["prostt5_embeddings"],
        )

    assert outputs["logits"].shape == (2, 6)
    print("Model with collated pooled batch test passed!")


if __name__ == "__main__":
    test_model_forward_pass()
    test_pooled_collate()
    test_model_with_collated_batch()
    print("\nAll tests passed!")
