"""Common utility functions for embedding extraction."""

import gzip
import os
from typing import List, Tuple

import h5py
import numpy as np
import torch


def get_device() -> str:
    """Get the best available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_text_maybe_gzip(file_path: str, mode: str = "rt"):
    """Open text file, handling gzip compression if needed."""
    if file_path.endswith(".gz"):
        return gzip.open(file_path, mode)
    return open(file_path, mode)


def create_id_emb_datasets(h5_file: h5py.File, embed_dim: int) -> Tuple[h5py.Dataset, h5py.Dataset]:
    """
    Create HDF5 datasets for protein IDs and embeddings.

    Args:
        h5_file: Open HDF5 file handle
        embed_dim: Embedding dimension

    Returns:
        ids_dataset: Variable-length string dataset for protein IDs
        emb_dataset: Float32 dataset for embeddings
    """
    dt_vlen_str = h5py.string_dtype(encoding='utf-8')

    ids_ds = h5_file.create_dataset(
        "ids",
        shape=(0,),
        maxshape=(None,),
        dtype=dt_vlen_str,
        chunks=True,
    )

    emb_ds = h5_file.create_dataset(
        "embeddings",
        shape=(0, embed_dim),
        maxshape=(None, embed_dim),
        dtype="float32",
        chunks=True,
    )

    return ids_ds, emb_ds


def append_id_emb_batch(
    ids_ds: h5py.Dataset,
    emb_ds: h5py.Dataset,
    protein_ids: List[str],
    embeddings: np.ndarray,
) -> int:
    """
    Append a batch of protein IDs and embeddings to HDF5 datasets.

    Args:
        ids_ds: HDF5 dataset for protein IDs
        emb_ds: HDF5 dataset for embeddings
        protein_ids: List of protein ID strings
        embeddings: Numpy array of shape (batch_size, embed_dim)

    Returns:
        New total count of entries
    """
    n_current = ids_ds.shape[0]
    n_new = len(protein_ids)
    n_total = n_current + n_new

    ids_ds.resize((n_total,))
    emb_ds.resize((n_total, emb_ds.shape[1]))

    ids_ds[n_current:n_total] = protein_ids
    emb_ds[n_current:n_total, :] = embeddings

    return n_total


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path:
        os.makedirs(path, exist_ok=True)
