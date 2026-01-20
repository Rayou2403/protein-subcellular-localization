#!/usr/bin/env python3
"""Concatenate (fuse) ESM-C and ProstT5 embeddings on common IDs.

Input formats supported:
  - 'flat' HDF5 with datasets: ids (vlen-str), embeddings (N, D)
  - 'per-key' HDF5 with one dataset per protein id (1D vector)

Output format:
  - HDF5 with datasets: ids (vlen-str), embeddings (N, D_esm + D_prost)
  - Attributes: embedding_dim, num_proteins, fusion_method
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import ensure_dir

def _load_esm(path: str) -> Dict[str, torch.Tensor]:
    out = {}
    with h5py.File(path, "r") as f:
        if "ids" in f and "embeddings" in f:
            ids = f["ids"][...]
            ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in ids]
            embs = f["embeddings"][...]
            for i, pid in enumerate(ids):
                out[pid] = torch.as_tensor(embs[i], dtype=torch.float32)
        else:
            for key in f.keys():
                arr = f[key][...]
                if getattr(arr, "dtype", None) is not None and arr.ndim == 1:
                    out[str(key)] = torch.as_tensor(arr, dtype=torch.float32)
    if not out:
        raise ValueError(f"No usable ESM embeddings in {path}")
    print(f"[ESMC] loaded {len(out)} vectors from {path}")
    return out

def _load_prost(path: str) -> Dict[str, torch.Tensor]:
    out = {}
    with h5py.File(path, "r") as f:
        ids = f["ids"][...]
        ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in ids]
        embs = f["embeddings"][...]
        for i, pid in enumerate(ids):
            out[pid] = torch.as_tensor(embs[i], dtype=torch.float32)
    print(f"[ProstT5] loaded {len(out)} vectors from {path}")
    return out

def run_fusion(esm_h5: str, prost_h5: str, output_h5: str) -> None:
    esm = _load_esm(esm_h5)
    prost = _load_prost(prost_h5)
    ids = sorted(set(esm.keys()) & set(prost.keys()))
    if not ids:
        raise ValueError("No common protein IDs between ESM-C and ProstT5 files.")
    fused = np.stack([(torch.cat([esm[i], prost[i]]).numpy()) for i in ids], axis=0)
    ensure_dir(os.path.dirname(output_h5) or ".")
    with h5py.File(output_h5, "w") as f:
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("ids", data=np.array(ids, dtype=object), dtype=dt)
        f.create_dataset("embeddings", data=fused, dtype=np.float32)
        f.attrs["embedding_dim"] = int(fused.shape[1])
        f.attrs["num_proteins"] = int(fused.shape[0])
        f.attrs["fusion_method"] = "concat[ESMC, ProstT5]"
    print(f"[Fused] {len(ids)} proteins → {output_h5} (dim={fused.shape[1]})")

def main():
    ap = argparse.ArgumentParser(description="Concatenate ESM-C and ProstT5 embeddings on common IDs.")
    ap.add_argument("--esm", required=True, help="ESM-C HDF5 (ids+embeddings or per-key)." )
    ap.add_argument("--prost", required=True, help="ProstT5 HDF5 (ids+embeddings)." )
    ap.add_argument("--output", required=True, help="Output HDF5 path." )
    args = ap.parse_args()
    run_fusion(args.esm, args.prost, args.output)

if __name__ == "__main__":
    main()
