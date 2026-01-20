#!/usr/bin/env python3
"""
ESM-C embedding extraction

Output format: HDF5 with datasets:
  - 'ids': vlen-string array of protein IDs
  - 'embeddings': float32 array of shape (N, D)
"""
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*_register_pytree_node.*deprecated.*",
    category=UserWarning,
    module=r"transformers\.utils\.generic",
)
warnings.filterwarnings(
    "ignore",
    message=".*_register_pytree_node.*deprecated.*",
    category=FutureWarning,
    module=r"transformers\.utils\.generic",
)
import argparse
import gc
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Iterable, List

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Bio import SeqIO

# ESMC SDK imports
try:
    from esm.models.esmc import ESMC
except ImportError as exc:
    raise ImportError(
        "ESM-C package not installed. Install with:\n"
        "  pip install git+https://github.com/evolutionaryscale/esm.git"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (
    get_device,
    open_text_maybe_gzip,
    create_id_emb_datasets,
    append_id_emb_batch,
    ensure_dir,
)

DEVICE = torch.device(get_device())

def _patch_esmc_tokenizer() -> None:
    """Add setters for special-token properties to avoid transformers setattr errors."""
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    except Exception:
        return

    def _setter_factory(name: str):
        def _setter(self, value):
            # SpecialTokensMixin initializes _special_tokens_map before setattr calls.
            try:
                token_map = object.__getattribute__(self, "_special_tokens_map")
            except AttributeError:
                token_map = None
            if isinstance(token_map, dict):
                token_map[name] = value
            else:
                self.__dict__[name] = value
        return _setter

    for name in ("cls_token", "pad_token", "mask_token", "eos_token"):
        prop = getattr(EsmSequenceTokenizer, name, None)
        if isinstance(prop, property) and prop.fset is None:
            setattr(EsmSequenceTokenizer, name, property(prop.fget, _setter_factory(name)))

    def _get_token_safe(self, token_name: str) -> str:
        token_str = None
        try:
            token_map = object.__getattribute__(self, "_special_tokens_map")
        except AttributeError:
            token_map = None
        if isinstance(token_map, dict):
            token_str = token_map.get(token_name)
        if not isinstance(token_str, str):
            token_str = self.__dict__.get(token_name)
        if not isinstance(token_str, str):
            raise AttributeError(f"Missing special token: {token_name}")
        return token_str

    EsmSequenceTokenizer._get_token = _get_token_safe

def _esmc_model_spec(model_name: str):
    if model_name == "esmc_300m":
        return {
            "d_model": 960,
            "n_heads": 15,
            "n_layers": 30,
            "root_key": "esmc-300",
            "weights_rel": "data/weights/esmc_300m_2024_12_v0.pth",
        }
    if model_name == "esmc_600m":
        return {
            "d_model": 1152,
            "n_heads": 18,
            "n_layers": 36,
            "root_key": "esmc-600",
            "weights_rel": "data/weights/esmc_600m_2024_12_v0.pth",
        }
    return None

def _load_state_dict(weights_path: Path, device: torch.device) -> dict:
    path_str = str(weights_path)
    try:
        return torch.load(path_str, map_location=device, weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path_str, map_location=device)

def _load_esmc_lowmem(model_name: str, device: torch.device) -> ESMC:
    from esm.tokenization import get_esmc_model_tokenizers
    from esm.utils.constants.esm3 import data_root

    spec = _esmc_model_spec(model_name)
    if spec is None:
        raise ValueError(f"Unsupported ESM-C model: {model_name}")

    with torch.device(device):
        model = ESMC(
            d_model=spec["d_model"],
            n_heads=spec["n_heads"],
            n_layers=spec["n_layers"],
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).eval()

    weights_path = data_root(spec["root_key"]) / spec["weights_rel"]
    state_dict = _load_state_dict(weights_path, device)
    try:
        # assign=True avoids an extra copy of weights (lower peak memory).
        model.load_state_dict(state_dict, assign=True)
    except TypeError:
        model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    return model

class ProteinDatasetESM(Dataset):
    def __init__(self, fasta_file: str):
        self.ids: List[str] = []
        self.sequences: List[str] = []
        with open_text_maybe_gzip(fasta_file, "rt") as fh:
            for rec in SeqIO.parse(fh, "fasta"):
                self.ids.append(rec.id.split("|")[0])
                self.sequences.append(str(rec.seq))

    def __len__(self): return len(self.ids)
    def __getitem__(self, idx): return self.sequences[idx], self.ids[idx]

def load_esmc(model_name: str = "esmc_300m") -> ESMC:
    _patch_esmc_tokenizer()
    if _esmc_model_spec(model_name) is not None:
        model = _load_esmc_lowmem(model_name, DEVICE)
    else:
        model = ESMC.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return model

def attention_pool(token_embeddings: torch.Tensor) -> torch.Tensor:
    """Dot-product attention pooling over length dimension.
    token_embeddings: (L, D) -> returns (D,)"""
    context = token_embeddings.mean(dim=0, keepdim=True)      # (1, D)
    scores = torch.matmul(token_embeddings, context.t()).squeeze(1)  # (L,)
    weights = torch.softmax(scores, dim=0).unsqueeze(1)       # (L, 1)
    return (token_embeddings * weights).sum(dim=0)            # (D,)

@torch.no_grad()
def _forward_embeddings_only(model: ESMC, sequence_tokens: torch.Tensor) -> torch.Tensor:
    """Forward pass without storing hidden states (lower memory)."""
    if getattr(model, "_use_flash_attn", False):
        raise RuntimeError("Low-memory path does not support flash attention.")
    sequence_id = sequence_tokens != model.tokenizer.pad_token_id
    x = model.embed(sequence_tokens)
    *batch_dims, _ = x.shape
    chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
    for block in model.transformer.blocks:
        x = block(x, sequence_id, None, None, chain_id)
    x = model.transformer.norm(x)
    return x

@torch.no_grad()
def extract_batch(model: ESMC, sequences: Iterable[str]) -> torch.Tensor:
    """Extract attention-pooled ESM-C embeddings for a batch of sequences.
    Returns tensor of shape (B, D).
    """
    out = []
    for seq in sequences:
        seq_tokens = model._tokenize([seq])  # (1, L)
        tok = _forward_embeddings_only(model, seq_tokens).squeeze(0)  # (L, D)
        pooled = attention_pool(tok)
        out.append(pooled.unsqueeze(0))
    return torch.cat(out, dim=0)

def run_esmc_extraction(fasta: str, output_h5: str, model_name: str = "esmc_300m",
                        batch_size: int = 32, num_workers: int = None) -> None:
    """Stream ESM-C embeddings to HDF5 (ids + embeddings)."""
    print(f"[ESMC] Device: {DEVICE}")
    model = load_esmc(model_name)
    ds = ProteinDatasetESM(fasta)
    ensure_dir(os.path.dirname(output_h5))
    if num_workers is None:
        num_workers = 0 if DEVICE.type == "cpu" else min(16, multiprocessing.cpu_count())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))

    # Prime first batch to get embedding dim
    it = iter(dl)
    try:
        first_seqs, first_ids = next(it)
    except StopIteration:
        raise ValueError("No sequences found in FASTA.")
    first_emb = extract_batch(model, first_seqs)
    D = int(first_emb.size(1))

    with h5py.File(output_h5, "w") as h5:
        ids_ds, emb_ds = create_id_emb_datasets(h5, D)
        n = append_id_emb_batch(ids_ds, emb_ds, list(first_ids), first_emb.cpu().numpy())
        for seqs, ids in tqdm(it, total=(len(dl)-1), desc="ESMC batches"):
            emb = extract_batch(model, seqs).cpu().numpy()
            n = append_id_emb_batch(ids_ds, emb_ds, list(ids), emb)

    print(f"[ESMC] Saved {n} embeddings to {output_h5} (dim={D}).")

def main():
    ap = argparse.ArgumentParser(description="Extract ESM-C embeddings to HDF5 (ids + embeddings)")
    ap.add_argument("--fasta", required=True, help="Input FASTA (.fa|.fasta[.gz])")
    ap.add_argument("--output", required=True, help="Output HDF5 path")
    ap.add_argument("--model", default="esmc_300m", help="ESM-C model name (default: esmc_300m)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: auto)")
    args = ap.parse_args()
    run_esmc_extraction(args.fasta, args.output, model_name=args.model,
                        batch_size=args.batch_size, num_workers=args.workers)

if __name__ == "__main__":
    main()
