#!/usr/bin/env python3
"""
ProtBert embedding extraction

Pooling modes:
  - 'cls': use the [CLS] token embedding.
  - 'meanpool': mean of amino-acid token embeddings (exclude [CLS]/[SEP]/padding).
  - 'both': produce two files, one per strategy.

Output format: HDF5 with datasets:
  - 'ids': vlen-string array of protein IDs
  - 'embeddings': float32 array of shape (N, H)
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
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Bio import SeqIO

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as exc:
    raise ImportError(
        "transformers not installed. Install with:\n"
        "  pip install transformers"
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


def _clean_and_space(seq: str) -> str:
    seq = re.sub(r"[UZOB]", "X", seq.upper().strip())
    return " ".join(list(seq))


def _iter_sorted(fasta: str) -> List[Tuple[str, str]]:
    """Return (processed_seq, id) sorted by raw AA length to reduce padding."""
    recs = []
    with open_text_maybe_gzip(fasta, "rt") as fh:
        for r in SeqIO.parse(fh, "fasta"):
            raw = str(r.seq)
            recs.append((len(raw), r.id.split("|")[0], _clean_and_space(raw)))
    recs.sort(key=lambda x: x[0])
    return [(p, i) for _, i, p in recs]


def _resolve_both_paths(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    if ext.lower() != ".h5":
        return f"{path}{suffix}.h5"
    return f"{root}{suffix}{ext}"


def _load_protbert():
    tok = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModel.from_pretrained("Rostlab/prot_bert", torch_dtype=dtype).to(DEVICE)
    model.eval()
    return tok, model


@torch.no_grad()
def _pool_cls(hidden: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0, :]


@torch.no_grad()
def _pool_mean(hidden: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor, cls_id: int, sep_id: int) -> torch.Tensor:
    mask = attn_mask.bool()
    mask = mask & (input_ids != cls_id) & (input_ids != sep_id)
    lengths = mask.sum(dim=1).clamp(min=1)
    masked = hidden * mask.unsqueeze(-1)
    return masked.sum(dim=1) / lengths.unsqueeze(-1)


@torch.no_grad()
def _embed_batch(tok, model, sequences: List[str], mode: str) -> torch.Tensor:
    enc = tok.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hidden = out.last_hidden_state
    if mode == "cls":
        pooled = _pool_cls(hidden)
    elif mode == "meanpool":
        pooled = _pool_mean(hidden, enc["input_ids"], enc["attention_mask"], tok.cls_token_id, tok.sep_token_id)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")
    return pooled.float()


def run_protbert_extraction(
    fasta: str,
    output_h5: str,
    batch_size: int = 4,
    pooling: str = "cls",
) -> List[str]:
    """Extract ProtBert embeddings. Returns list of output file paths written."""
    modes = [pooling] if pooling != "both" else ["cls", "meanpool"]
    outputs = []
    data = _iter_sorted(fasta)
    if not data:
        raise ValueError("No sequences in FASTA.")

    tok, model = _load_protbert()

    dl = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    first_seqs, first_ids = next(iter(dl))
    for mode in modes:
        out_path = output_h5
        if pooling == "both":
            suf = "-cls" if mode == "cls" else "-meanpool"
            out_path = _resolve_both_paths(output_h5, suf)
        first_emb = _embed_batch(tok, model, list(first_seqs), mode)
        H = int(first_emb.size(1))
        with h5py.File(out_path, "w") as h5:
            ids_ds, emb_ds = create_id_emb_datasets(h5, H)
            n = append_id_emb_batch(ids_ds, emb_ds, list(first_ids), first_emb.cpu().numpy())
            it = iter(dl)
            next(it)
            for seqs, ids in tqdm(it, total=(len(dl) - 1), desc=f"ProtBert[{mode}] batches"):
                embs = _embed_batch(tok, model, list(seqs), mode).cpu().numpy()
                n = append_id_emb_batch(ids_ds, emb_ds, list(ids), embs)
        print(f"[ProtBert:{mode}] Saved {n} embeddings to {out_path} (dim={H}).")
        outputs.append(out_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main():
    ap = argparse.ArgumentParser(description="Extract ProtBert embeddings to HDF5 (ids + embeddings)")
    ap.add_argument("--fasta", required=True, help="Input FASTA (.fa|.fasta[.gz])")
    ap.add_argument("--output", required=True, help="Output HDF5 path. If --pooling both, acts as a prefix.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--pooling", choices=["cls", "meanpool", "both"], default="cls")
    args = ap.parse_args()
    run_protbert_extraction(args.fasta, args.output, batch_size=args.batch_size, pooling=args.pooling)


if __name__ == "__main__":
    main()
