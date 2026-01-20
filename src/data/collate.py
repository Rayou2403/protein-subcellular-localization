"""
Custom collate functions for protein datasets.
"""

from typing import List, Dict
import torch


def collate_variable_length(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate pooled (fixed-size) embeddings.

    Args:
        batch: List of samples from ProteinLocalizationDataset

    Returns:
        Dictionary with:
            - esmc_embeddings: (batch_size, esmc_dim)
            - prostt5_embeddings: (batch_size, prostt5_dim)
            - labels: (batch_size,)
            - protein_ids: List[str]
    """
    esmc_embs = torch.stack([item["esmc_embedding"] for item in batch])
    prostt5_embs = torch.stack([item["prostt5_embedding"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    protein_ids = [item["protein_id"] for item in batch]

    return {
        "esmc_embeddings": esmc_embs,
        "prostt5_embeddings": prostt5_embs,
        "labels": labels,
        "protein_ids": protein_ids,
    }
