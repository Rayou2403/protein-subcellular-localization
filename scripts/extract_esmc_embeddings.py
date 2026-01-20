"""
ESM-C embedding extraction (wrapper).

Use this wrapper for backward compatibility with the README.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embeddings.fetch_esmc import run_esmc_extraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ESM-C embeddings to HDF5 file"
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--model",
        default="esmc_300m",
        help="ESM-C model name (default: esmc_300m)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers (default: auto)",
    )
    args = parser.parse_args()

    run_esmc_extraction(
        args.fasta,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
