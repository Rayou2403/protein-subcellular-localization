#!/usr/bin/env python3
"""
ProstT5 embedding extraction

Pooling modes:
  - 'aa2fold': take hidden state of the first token (<AA2fold>), CLS-like.
  - 'meanpool': mean of amino-acid token embeddings (exclude <AA2fold> and padding).
  - 'both': produce two files, one per strategy.

Output format per run: HDF5 with datasets:
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
warnings.filterwarnings(
    "ignore",
    message="You are using the default legacy behaviour.*T5Tokenizer.*",
    category=UserWarning,
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
    from transformers import T5Tokenizer, T5EncoderModel
except ImportError as exc:
    raise ImportError(
        "transformers not installed. Install with:\n"
        "  pip install transformers sentencepiece"
    ) from exc
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (
    get_device,
    open_text_maybe_gzip,
    create_id_emb_datasets,
    append_id_emb_batch,
    ensure_dir,
)

DEVICE = get_device()

def _clean_and_space(seq: str) -> str:
    seq = re.sub(r"[UZOB]", "X", seq.upper().strip())
    return " ".join(list(seq))

def _prep(seq: str) -> str:
    return "<AA2fold> " + _clean_and_space(seq)

def _load_prost(offload_dir: str | None = None, max_memory: str | None = None):
    tok = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    kwargs = {
        "use_safetensors": False,
        "torch_dtype": dtype,
    }
    if offload_dir:
        try:
            import accelerate  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "ProstT5 offload requires accelerate. Install with:\n"
                "  pip install accelerate"
            ) from exc
        ensure_dir(offload_dir)
        device_map = "cpu" if DEVICE == "cpu" else "auto"
        kwargs.update({
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "offload_folder": offload_dir,
            "offload_state_dict": True,
        })
        if max_memory:
            kwargs["max_memory"] = {"cpu": max_memory}
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5", **kwargs)
    else:
        kwargs["low_cpu_mem_usage"] = False
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5", **kwargs).to(DEVICE)

    model.eval()
    return tok, model

@torch.no_grad()
def _pool_aa2fold(hidden: torch.Tensor) -> torch.Tensor:
    # hidden: (B, L, H)
    return hidden[:, 0, :]  # <AA2fold> position

@torch.no_grad()
def _pool_mean(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # hidden: (B, L, H); attn_mask: (B, L) with 1 for valid tokens
    B, L, H = hidden.shape
    out = []
    for i in range(B):
        valid_len = int(attn_mask[i].sum().item()) - 1  # exclude prefix
        if valid_len > 0:
            aa_tokens = hidden[i, 1:1+valid_len, :]
            out.append(aa_tokens.mean(dim=0))
        else:
            out.append(torch.zeros(H, device=hidden.device))
    return torch.stack(out, dim=0)

@torch.no_grad()
def _embed_batch(tok, model, sequences: List[str], mode: str, input_device: str) -> torch.Tensor:
    enc = tok.batch_encode_plus(
        sequences, add_special_tokens=True, padding="longest",
        truncation=True, max_length=1024, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(input_device)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hidden = out.last_hidden_state  # (B, L, H)
    if mode == "aa2fold":
        pooled = _pool_aa2fold(hidden)
    elif mode == "meanpool":
        pooled = _pool_mean(hidden, enc["attention_mask"])
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")
    return pooled.float()

def _iter_sorted(fasta: str) -> List[Tuple[str, str]]:
    """Return (processed_seq, id) sorted by raw AA length to reduce padding."""
    recs = []
    with open_text_maybe_gzip(fasta, "rt") as fh:
        for r in SeqIO.parse(fh, "fasta"):
            raw = str(r.seq)
            recs.append((len(raw), r.id.split("|")[0], _prep(raw)))
    recs.sort(key=lambda x: x[0])
    return [(p, i) for _, i, p in recs]

def _resolve_both_paths(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    if ext.lower() != ".h5":
        # if no .h5, just append suffix
        return f"{path}{suffix}.h5"
    return f"{root}{suffix}{ext}"

def run_prost_extraction(
    fasta: str,
    output_h5: str,
    batch_size: int = 32,
    pooling: str = "aa2fold",
    offload_dir: str | None = None,
    max_memory: str | None = None,
) -> List[str]:
    """Extract ProstT5 embeddings. Returns list of output file paths written.

    If pooling == 'both', writes two files: <output>[-aa2fold].h5 and <output>[-meanpool].h5
    """
    modes = [pooling] if pooling != "both" else ["aa2fold", "meanpool"]
    outputs = []
    data = _iter_sorted(fasta)
    if not data:
        raise ValueError("No sequences in FASTA.")
    tok, model = _load_prost(offload_dir=offload_dir, max_memory=max_memory)
    input_device = "cpu" if offload_dir else DEVICE

    # Peek one batch to infer dim per mode (can differ theoretically)
    dl = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    first_seqs, first_ids = next(iter(dl))
    for mode in modes:
        # resolve path per mode if 'both'
        out_path = output_h5
        if pooling == "both":
            suf = "-aa2fold" if mode == "aa2fold" else "-meanpool"
            out_path = _resolve_both_paths(output_h5, suf)

        first_emb = _embed_batch(tok, model, list(first_seqs), mode, input_device)
        H = int(first_emb.size(1))
        n = 0
        with h5py.File(out_path, "w") as h5:
            ids_ds, emb_ds = create_id_emb_datasets(h5, H)
            n = append_id_emb_batch(ids_ds, emb_ds, list(first_ids), first_emb.cpu().numpy())
            it = iter(dl); next(it)  # skip first
            for seqs, ids in tqdm(it, total=(len(dl)-1), desc=f"ProstT5[{mode}] batches"):
                embs = _embed_batch(tok, model, list(seqs), mode, input_device).cpu().numpy()
                n = append_id_emb_batch(ids_ds, emb_ds, list(ids), embs)
        print(f"[ProstT5:{mode}] Saved {n} embeddings to {out_path} (dim={H}).")
        outputs.append(out_path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs

def main():
    ap = argparse.ArgumentParser(description="Extract ProstT5 embeddings to HDF5 (ids + embeddings)")
    ap.add_argument("--fasta", required=True, help="Input FASTA (.fa|.fasta[.gz])")
    ap.add_argument("--output", required=True, help="Output HDF5 path. If --pooling both, acts as a prefix.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--pooling", choices=["aa2fold", "meanpool", "both"], default="aa2fold")
    args = ap.parse_args()
    run_prost_extraction(args.fasta, args.output, batch_size=args.batch_size, pooling=args.pooling)

if __name__ == "__main__":
    main()
