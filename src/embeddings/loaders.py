"""
Utilities for loading precomputed embeddings.
"""

import os
from typing import Dict, Optional
import torch
import numpy as np


class EmbeddingCache:
    """
    Loads and caches protein embeddings from disk.

    Supports both .pt (PyTorch) and .npy (NumPy) formats.
    """

    def __init__(
        self,
        esmc_dir: str,
        prostt5_dir: str,
        cache_in_memory: bool = False,
    ):
        """
        Initialize embedding cache.

        Args:
            esmc_dir: Directory with ESM-C embeddings
            prostt5_dir: Directory with ProstT5 embeddings
            cache_in_memory: If True, load all embeddings into RAM
        """
        self.esmc_dir = esmc_dir
        self.prostt5_dir = prostt5_dir
        self.cache_in_memory = cache_in_memory

        self._esmc_cache: Dict[str, torch.Tensor] = {}
        self._prostt5_cache: Dict[str, torch.Tensor] = {}

        if cache_in_memory:
            print("Loading all embeddings into memory...")
            self._preload_embeddings()

    def _preload_embeddings(self):
        """Load all embeddings into memory cache."""
        esmc_files = [f for f in os.listdir(self.esmc_dir) if f.endswith((".pt", ".npy"))]
        prostt5_files = [f for f in os.listdir(self.prostt5_dir) if f.endswith((".pt", ".npy"))]

        for fname in esmc_files:
            protein_id = fname.replace(".pt", "").replace(".npy", "")
            self._esmc_cache[protein_id] = self.load_embedding(
                os.path.join(self.esmc_dir, fname)
            )

        for fname in prostt5_files:
            protein_id = fname.replace(".pt", "").replace(".npy", "")
            self._prostt5_cache[protein_id] = self.load_embedding(
                os.path.join(self.prostt5_dir, fname)
            )

        print(f"Loaded {len(self._esmc_cache)} ESM-C and {len(self._prostt5_cache)} ProstT5 embeddings")

    def load_embedding(self, path: str) -> torch.Tensor:
        """
        Load a single embedding file.

        Args:
            path: Path to .pt or .npy file

        Returns:
            Tensor of shape (seq_len, dim)
        """
        if path.endswith(".pt"):
            return torch.load(path, map_location="cpu")
        elif path.endswith(".npy"):
            return torch.from_numpy(np.load(path))
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def get_esmc_embedding(self, protein_id: str) -> torch.Tensor:
        """Get ESM-C embedding for a protein."""
        if self.cache_in_memory:
            return self._esmc_cache[protein_id]
        else:
            path = os.path.join(self.esmc_dir, f"{protein_id}.pt")
            if not os.path.exists(path):
                path = os.path.join(self.esmc_dir, f"{protein_id}.npy")
            return self.load_embedding(path)

    def get_prostt5_embedding(self, protein_id: str) -> torch.Tensor:
        """Get ProstT5 embedding for a protein."""
        if self.cache_in_memory:
            return self._prostt5_cache[protein_id]
        else:
            path = os.path.join(self.prostt5_dir, f"{protein_id}.pt")
            if not os.path.exists(path):
                path = os.path.join(self.prostt5_dir, f"{protein_id}.npy")
            return self.load_embedding(path)
