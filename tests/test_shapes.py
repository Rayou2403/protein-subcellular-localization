"""
Unit tests for model forward pass and data pipeline.
"""

import sys
from pathlib import Path
import torch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DualEmbeddingFusionModel
from src.data.collate import collate_variable_length


def test_model_forward_pass():
    """Test model forward pass with dummy data."""
    batch_size = 4
    seq_len = 50
    esmc_dim = 1280
    prostt5_dim = 1024
    hidden_dim = 512
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
    esmc_emb = torch.randn(batch_size, seq_len, esmc_dim)
    prostt5_emb = torch.randn(batch_size, seq_len, prostt5_dim)
    esmc_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    prostt5_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(esmc_emb, prostt5_emb, esmc_mask, prostt5_mask)

    # Check shapes
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, num_classes)
    assert "gate_weights" in outputs

    print("Model forward pass test passed!")


def test_variable_length_collate():
    """Test collate function with variable-length sequences."""
    # Create dummy batch
    batch = [
        {
            "esmc_embedding": torch.randn(30, 1280),
            "prostt5_embedding": torch.randn(30, 1024),
            "label": torch.tensor(0),
            "protein_id": "P001",
        },
        {
            "esmc_embedding": torch.randn(50, 1280),
            "prostt5_embedding": torch.randn(50, 1024),
            "label": torch.tensor(1),
            "protein_id": "P002",
        },
        {
            "esmc_embedding": torch.randn(20, 1280),
            "prostt5_embedding": torch.randn(20, 1024),
            "label": torch.tensor(2),
            "protein_id": "P003",
        },
    ]

    # Collate
    collated = collate_variable_length(batch)

    # Check shapes
    assert collated["esmc_embeddings"].shape == (3, 50, 1280)
    assert collated["prostt5_embeddings"].shape == (3, 50, 1024)
    assert collated["esmc_mask"].shape == (3, 50)
    assert collated["prostt5_mask"].shape == (3, 50)
    assert collated["labels"].shape == (3,)
    assert len(collated["protein_ids"]) == 3

    # Check masks
    assert collated["esmc_mask"][0, :30].all()
    assert not collated["esmc_mask"][0, 30:].any()
    assert collated["esmc_mask"][1, :50].all()
    assert collated["esmc_mask"][2, :20].all()
    assert not collated["esmc_mask"][2, 20:].any()

    print("Variable-length collate test passed!")


def test_model_with_collated_batch():
    """Test model with collated variable-length batch."""
    batch = [
        {
            "esmc_embedding": torch.randn(30, 1280),
            "prostt5_embedding": torch.randn(30, 1024),
            "label": torch.tensor(0),
            "protein_id": "P001",
        },
        {
            "esmc_embedding": torch.randn(50, 1280),
            "prostt5_embedding": torch.randn(50, 1024),
            "label": torch.tensor(1),
            "protein_id": "P002",
        },
    ]

    collated = collate_variable_length(batch)

    model = DualEmbeddingFusionModel(
        esmc_dim=1280,
        prostt5_dim=1024,
        hidden_dim=512,
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
            collated["esmc_mask"],
            collated["prostt5_mask"],
        )

    assert outputs["logits"].shape == (2, 6)
    print("Model with collated batch test passed!")


if __name__ == "__main__":
    test_model_forward_pass()
    test_variable_length_collate()
    test_model_with_collated_batch()
    print("\nAll tests passed!")
