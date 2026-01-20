"""
Create a small, balanced subset from a splits CSV and export a FASTA file.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small dataset subset from an existing splits CSV"
    )
    parser.add_argument(
        "--splits",
        required=True,
        help="Path to CSV with split column (e.g., data/splits.csv)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for small dataset (e.g., data/small)",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=50,
        help="Max samples per class per split (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    return parser.parse_args()


def sample_per_split(
    df: pd.DataFrame,
    split: str,
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    split_df = df[df["split"] == split]
    if split_df.empty:
        return split_df

    sampled = []
    for label, group in split_df.groupby("label"):
        if len(group) <= max_per_class:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=max_per_class, random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def write_fasta(df: pd.DataFrame, fasta_path: Path) -> None:
    required = {"protein_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for FASTA: {missing_str}")

    has_label = "label" in df.columns
    has_gram = "gram_type" in df.columns
    has_partition = "partition" in df.columns

    with open(fasta_path, "w") as fh:
        for row in df.itertuples(index=False):
            header_parts = [str(getattr(row, "protein_id"))]
            if has_label:
                header_parts.append(str(getattr(row, "label")))
            if has_gram:
                header_parts.append(str(getattr(row, "gram_type")))
            if has_partition:
                header_parts.append(str(getattr(row, "partition")))
            header = "|".join(header_parts)
            seq = str(getattr(row, "sequence"))
            fh.write(f">{header}\n{seq}\n")


def main() -> None:
    args = parse_args()

    if args.max_per_class < 1:
        raise ValueError("--max_per_class must be >= 1")

    df = pd.read_csv(args.splits)
    if "split" not in df.columns:
        raise ValueError("Input CSV must include a 'split' column. Run prepare_splits.py first.")
    if "label" not in df.columns:
        raise ValueError("Input CSV must include a 'label' column.")

    splits = [s for s in ["train", "val", "test"] if s in df["split"].unique()]
    if not splits:
        splits = sorted(df["split"].unique())

    subset_frames = [
        sample_per_split(df, split, args.max_per_class, args.seed)
        for split in splits
    ]
    subset = pd.concat(subset_frames, ignore_index=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "splits.csv"
    subset.to_csv(out_csv, index=False)

    fasta_path = out_dir / "sequences.fasta"
    write_fasta(subset, fasta_path)

    print(f"Saved {len(subset)} rows to {out_csv}")
    print(f"Saved FASTA to {fasta_path}")
    print("\nSubset sizes by split:")
    print(subset["split"].value_counts())
    print("\nSubset sizes by class:")
    print(subset["label"].value_counts())


if __name__ == "__main__":
    main()
