"""
ProstT5 embedding extraction (wrapper).

Use this wrapper for backward compatibility with the README.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embeddings.fetch_prostt5 import run_prost_extraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ProstT5 embeddings to HDF5 file"
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
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--pooling",
        choices=["aa2fold", "meanpool", "both"],
        default="aa2fold",
        help="Pooling mode (default: aa2fold)",
    )
    args = parser.parse_args()

    run_prost_extraction(
        args.fasta,
        args.output,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )


if __name__ == "__main__":
    main()
