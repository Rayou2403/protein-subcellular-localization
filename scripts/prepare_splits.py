"""
Prepare train/val/test splits with sequence identity clustering.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splits import create_identity_based_splits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create leakage-safe train/val/test splits"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to input metadata CSV [protein_id, sequence, label]",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV with added 'split' column",
    )
    parser.add_argument(
        "--identity_threshold",
        type=float,
        default=0.3,
        help="Sequence identity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation set fraction (default: 0.1)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test set fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--use_mmseqs",
        action="store_true",
        help="Use MMseqs2 for clustering (requires installation)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Creating train/val/test splits...")
    print(f"Input: {args.metadata}")
    print(f"Output: {args.output}")
    print(f"Identity threshold: {args.identity_threshold}")
    print(f"Val size: {args.val_size}")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    print(f"Use MMseqs2: {args.use_mmseqs}")

    create_identity_based_splits(
        metadata_path=args.metadata,
        output_path=args.output,
        identity_threshold=args.identity_threshold,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        use_mmseqs=args.use_mmseqs,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
