#!/usr/bin/env python3
"""
Orchestrate extraction of ESM-C, ProstT5 (with pooling), and fusion.

By using --prost_pooling of both, you will generate two output files with -aa2fold and -meanpool suffixes.

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
import os
import random

from Bio import SeqIO

from .fetch_esmc import run_esmc_extraction
from .fetch_prostt5 import run_prost_extraction
from .concat import run_fusion
from src.utils.common import ensure_dir, open_text_maybe_gzip

def _suffix(path: str, tag: str) -> str:
    root, ext = os.path.splitext(path)
    if ext.lower() != ".h5":
        return f"{path}-{tag}.h5"
    return f"{root}-{tag}{ext}"

def _subset_output_path(input_path: str, frac: float | None, max_seqs: int | None) -> str:
    base = os.path.basename(input_path)
    if base.endswith(".gz"):
        base = base[:-3]
    tag_parts = []
    if frac is not None:
        tag_parts.append(f"frac{frac:g}")
    if max_seqs is not None:
        tag_parts.append(f"max{max_seqs}")
    tag = "-".join(tag_parts) if tag_parts else "subset"
    return os.path.join("data", "subsets", f"{base}.{tag}.fasta")

def _write_subset_fasta(input_path: str, output_path: str, frac: float | None,
                        max_seqs: int | None, seed: int) -> str:
    if frac is not None and not (0.0 < frac <= 1.0):
        raise SystemExit("--subset_frac must be in (0, 1].")
    ensure_dir(os.path.dirname(output_path))
    rng = random.Random(seed)
    n_in = 0
    n_out = 0
    with open_text_maybe_gzip(input_path, "rt") as fin, open(output_path, "wt") as fout:
        for rec in SeqIO.parse(fin, "fasta"):
            n_in += 1
            take = True
            if frac is not None:
                take = rng.random() < frac
            if take:
                SeqIO.write(rec, fout, "fasta")
                n_out += 1
                if max_seqs is not None and n_out >= max_seqs:
                    break
    if n_out == 0:
        raise SystemExit("Subset selection produced zero sequences.")
    print(f"[Subset] Wrote {n_out} / {n_in} sequences to {output_path}")
    return output_path

def main():
    ap = argparse.ArgumentParser(description="Orchestrate ESM-C/ProstT5 extraction and fusion.")
    # ESM-C
    ap.add_argument("--esm_fasta", help="Input FASTA for ESM-C extraction (optional if --esm_file provided)")
    ap.add_argument("--esm_file", help="Existing ESM-C HDF5 file to reuse (optional)")
    ap.add_argument("--esm_out", default="esmc_embeddings.h5", help="Output HDF5 for ESM-C (if extracting)")
    ap.add_argument("--esm_model", default="esmc_300m", help="ESM-C model name (default: esmc_300m)")
    ap.add_argument("--esm_batch", type=int, default=32, help="Batch size for ESM-C token→embedding loop")        
    # ProstT5
    ap.add_argument("--prost_fasta", help="Input FASTA for ProstT5 extraction (defaults to --esm_fasta if omitted and that is provided)")
    ap.add_argument("--prost_file", help="Existing ProstT5 HDF5 to reuse (optional)")
    ap.add_argument("--prost_out", default="prost_embeddings.h5", help="Output HDF5 for ProstT5 (if extracting). If --prost_pooling both, acts as prefix.")
    ap.add_argument("--prost_pooling", choices=["aa2fold", "meanpool", "both"], default="aa2fold")
    ap.add_argument("--prost_batch", type=int, default=32, help="Batch size for ProstT5 extraction")        
    # Fusion
    ap.add_argument("--fused_out", help="Output HDF5 for fused embeddings. If --prost_pooling both, will emit two files with -aa2fold/-meanpool suffixes.")
    # Subset
    ap.add_argument("--subset_frac", type=float, help="Sample fraction of sequences (e.g. 0.1 for 10%)")
    ap.add_argument("--subset_max", type=int, help="Max sequences to keep (optional cap)")
    ap.add_argument("--subset_seed", type=int, default=42, help="RNG seed for subset sampling")
    ap.add_argument("--subset_out", help="Output FASTA for subset (optional; defaults to data/subsets/...)")
    args = ap.parse_args()

    # Default prost_fasta to esm_fasta if not provided and no prost_file given
    if not args.prost_fasta and not args.prost_file and args.esm_fasta:
        args.prost_fasta = args.esm_fasta

    # Optional subset selection (keeps ESM/Prost aligned)
    if (args.subset_frac is not None or args.subset_max is not None) and args.esm_fasta:
        if args.prost_fasta and args.prost_fasta != args.esm_fasta:
            raise SystemExit("Subset sampling requires the same FASTA for ESM-C and ProstT5.")
        out_path = args.subset_out or _subset_output_path(args.esm_fasta, args.subset_frac, args.subset_max)
        subset_path = _write_subset_fasta(
            args.esm_fasta,
            out_path,
            frac=args.subset_frac,
            max_seqs=args.subset_max,
            seed=args.subset_seed,
        )
        args.esm_fasta = subset_path
        args.prost_fasta = subset_path

    # ESM path
    esm_path = args.esm_file
    if args.esm_fasta:
        esm_path = args.esm_out
        run_esmc_extraction(args.esm_fasta, esm_path, model_name=args.esm_model, batch_size=args.esm_batch)
    if not esm_path:
        raise SystemExit("Provide either --esm_fasta or --esm_file.")

    # Prost path(s)
    prost_paths = []
    if args.prost_fasta:
        prost_paths = run_prost_extraction(args.prost_fasta, args.prost_out,
                                           batch_size=args.prost_batch, pooling=args.prost_pooling)
    elif args.prost_file:
        prost_paths = [args.prost_file]
    else:
        raise SystemExit("Provide either --prost_fasta (or implicitly via --esm_fasta) or --prost_file.")

    # Fusion (optional but common)
    if args.fused_out:
        if len(prost_paths) == 1:
            run_fusion(esm_path, prost_paths[0], args.fused_out)
        else:
            # we have both aa2fold + meanpool; map suffixes
            for p in prost_paths:
                tag = "aa2fold" if p.endswith("-aa2fold.h5") else ("meanpool" if p.endswith("-meanpool.h5") else "prost")
                out = _suffix(args.fused_out, tag)
                run_fusion(esm_path, p, out)

if __name__ == "__main__":
    main()
