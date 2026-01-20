"""
PyTorch Dataset for protein localization with dual embeddings.

Supports loading embeddings from HDF5 files (pooled, fixed-size vectors).
"""

from typing import Dict, Optional, List
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ProteinLocalizationDataset(Dataset):
    """
    Dataset for protein subcellular localization prediction.

    Loads precomputed ESM-C and ProstT5 embeddings from HDF5 files.
    Embeddings are pooled (fixed-size vectors per protein).
    """

    def __init__(
        self,
        metadata_path: str,
        esmc_h5_path: str,
        prostt5_h5_path: str,
        split: str = "train",
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize dataset.

        Args:
            metadata_path: Path to CSV with columns [protein_id, sequence, label, split]
            esmc_h5_path: Path to HDF5 file with ESM-C embeddings
            prostt5_h5_path: Path to HDF5 file with ProstT5 embeddings
            split: One of 'train', 'val', 'test'
            label_to_idx: Optional mapping from label strings to indices
        """
        self.split = split

        # Load embeddings from HDF5 into memory (they're pooled, so small)
        self.esmc_embeddings = self._load_h5_embeddings(esmc_h5_path)
        self.prostt5_embeddings = self._load_h5_embeddings(prostt5_h5_path)

        # Load metadata
        df = pd.read_csv(metadata_path)

        # Filter by split
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        # Filter to only proteins with both embeddings present
        df = self._filter_missing_embeddings(df)
        self.metadata = df

        # Create label mapping
        if label_to_idx is None:
            unique_labels = sorted(df["label"].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        # Get embedding dimensions
        sample_id = self.metadata.iloc[0]["protein_id"]
        self.esmc_dim = self.esmc_embeddings[sample_id].shape[0]
        self.prostt5_dim = self.prostt5_embeddings[sample_id].shape[0]

    def _load_h5_embeddings(self, h5_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from HDF5 file into a dictionary."""
        embeddings = {}
        with h5py.File(h5_path, "r") as f:
            ids = f["ids"][...]
            embs = f["embeddings"][...]

            for i, pid in enumerate(ids):
                # Handle bytes vs string
                if isinstance(pid, bytes):
                    pid = pid.decode("utf-8")
                embeddings[str(pid)] = embs[i]

        print(f"Loaded {len(embeddings)} embeddings from {h5_path}")
        return embeddings

    def _filter_missing_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out proteins that don't have both embeddings."""
        original_count = len(df)
        valid_indices = []

        for idx, row in df.iterrows():
            protein_id = str(row["protein_id"])
            if protein_id in self.esmc_embeddings and protein_id in self.prostt5_embeddings:
                valid_indices.append(idx)

        filtered_df = df.loc[valid_indices].reset_index(drop=True)
        filtered_count = len(filtered_df)

        if filtered_count < original_count:
            print(f"Filtered {original_count - filtered_count} proteins with missing embeddings "
                  f"({filtered_count}/{original_count} remaining)")

        if filtered_count == 0:
            raise ValueError(
                f"No proteins have both embeddings! "
                f"ESM-C has {len(self.esmc_embeddings)} proteins, "
                f"ProstT5 has {len(self.prostt5_embeddings)} proteins."
            )

        return filtered_df

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
                - esmc_embedding: (esmc_dim,)
                - prostt5_embedding: (prostt5_dim,)
                - label: int
                - protein_id: str
        """
        row = self.metadata.iloc[idx]
        protein_id = str(row["protein_id"])
        label_str = row["label"]
        label_idx = self.label_to_idx[label_str]

        # Get embeddings from memory
        esmc_emb = torch.from_numpy(self.esmc_embeddings[protein_id]).float()
        prostt5_emb = torch.from_numpy(self.prostt5_embeddings[protein_id]).float()

        return {
            "esmc_embedding": esmc_emb,
            "prostt5_embedding": prostt5_emb,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "protein_id": protein_id,
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency class weights for handling imbalance.

        Returns:
            Tensor of shape (num_classes,)
        """
        label_counts = self.metadata["label"].value_counts()
        weights = torch.zeros(self.num_classes)

        total = len(self.metadata)
        for label_str, count in label_counts.items():
            idx = self.label_to_idx[label_str]
            weights[idx] = total / (self.num_classes * count)

        return weights

    def get_sample_weights(self) -> List[float]:
        """
        Get per-sample weights for balanced sampling.

        Returns:
            List of weights, one per sample
        """
        class_weights = self.get_class_weights()
        sample_weights = []

        for label_str in self.metadata["label"]:
            idx = self.label_to_idx[label_str]
            sample_weights.append(class_weights[idx].item())

        return sample_weights
