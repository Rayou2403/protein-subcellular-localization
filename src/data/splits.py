"""
Leakage-safe data splitting using sequence identity clustering.
"""

import subprocess
import tempfile
import os
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_identity_based_splits(
    metadata_path: str,
    output_path: str,
    identity_threshold: float = 0.3,
    val_size: float = 0.1,
    test_size: float = 0.15,
    seed: int = 42,
    use_mmseqs: bool = False,
) -> pd.DataFrame:
    """
    Create train/val/test splits ensuring no high-identity sequences across splits.

    This prevents data leakage from near-duplicate proteins. Sequences with
    identity >= identity_threshold are kept in the same split.

    Args:
        metadata_path: Path to input CSV with [protein_id, sequence, label]
        output_path: Path to save output CSV with added 'split' column
        identity_threshold: Sequence identity threshold (0.3 = 30%)
        val_size: Fraction for validation
        test_size: Fraction for test
        seed: Random seed
        use_mmseqs: If True, use MMseqs2 clustering (requires installation)
                   If False, use simple random splitting (fallback)

    Returns:
        DataFrame with added 'split' column
    """
    np.random.seed(seed)

    df = pd.read_csv(metadata_path)

    if use_mmseqs and _mmseqs_available():
        print("Using MMseqs2 for sequence identity clustering...")
        clusters = _cluster_sequences_mmseqs(df, identity_threshold)
        df = _assign_splits_by_cluster(df, clusters, val_size, test_size, seed)
    else:
        print("MMseqs2 not available. Using stratified random split...")
        print("Warning: This may not prevent sequence identity leakage.")
        df = _simple_stratified_split(df, val_size, test_size, seed)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Splits saved to {output_path}")
    print(f"Train: {(df['split'] == 'train').sum()}")
    print(f"Val: {(df['split'] == 'val').sum()}")
    print(f"Test: {(df['split'] == 'test').sum()}")

    return df


def _mmseqs_available() -> bool:
    """Check if MMseqs2 is available."""
    try:
        subprocess.run(
            ["mmseqs", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _cluster_sequences_mmseqs(
    df: pd.DataFrame,
    identity_threshold: float,
) -> Dict[str, int]:
    """
    Cluster sequences by identity using MMseqs2.

    Returns:
        Dictionary mapping protein_id to cluster_id
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sequences to FASTA
        fasta_path = os.path.join(tmpdir, "sequences.fasta")
        with open(fasta_path, "w") as f:
            for _, row in df.iterrows():
                f.write(f">{row['protein_id']}\n{row['sequence']}\n")

        # Run MMseqs2 clustering
        db_path = os.path.join(tmpdir, "seqDB")
        clu_path = os.path.join(tmpdir, "cluDB")
        tsv_path = os.path.join(tmpdir, "clusters.tsv")

        subprocess.run(
            ["mmseqs", "createdb", fasta_path, db_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        subprocess.run(
            [
                "mmseqs",
                "cluster",
                db_path,
                clu_path,
                tmpdir,
                "--min-seq-id",
                str(identity_threshold),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        subprocess.run(
            ["mmseqs", "createtsv", db_path, db_path, clu_path, tsv_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Parse clusters
        clusters = {}
        cluster_id = 0
        cluster_map = {}

        with open(tsv_path) as f:
            for line in f:
                representative, member = line.strip().split("\t")

                if representative not in cluster_map:
                    cluster_map[representative] = cluster_id
                    cluster_id += 1

                clusters[member] = cluster_map[representative]

        return clusters


def _assign_splits_by_cluster(
    df: pd.DataFrame,
    clusters: Dict[str, int],
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    """
    Assign splits ensuring entire clusters stay together.
    """
    # Add cluster column
    df["cluster"] = df["protein_id"].map(clusters)

    # Get unique clusters with their class distributions
    cluster_labels = (
        df.groupby("cluster")["label"]
        .apply(lambda x: x.mode()[0])  # Majority label per cluster
        .to_dict()
    )

    unique_clusters = list(cluster_labels.keys())
    cluster_label_list = [cluster_labels[c] for c in unique_clusters]

    # Split clusters
    train_clusters, test_clusters = train_test_split(
        unique_clusters,
        test_size=test_size,
        stratify=cluster_label_list,
        random_state=seed,
    )

    train_labels = [cluster_labels[c] for c in train_clusters]
    train_clusters, val_clusters = train_test_split(
        train_clusters,
        test_size=val_size / (1 - test_size),
        stratify=train_labels,
        random_state=seed,
    )

    # Assign splits
    train_set = set(train_clusters)
    val_set = set(val_clusters)
    test_set = set(test_clusters)

    def assign_split(cluster):
        if cluster in train_set:
            return "train"
        elif cluster in val_set:
            return "val"
        else:
            return "test"

    df["split"] = df["cluster"].apply(assign_split)
    df = df.drop(columns=["cluster"])

    return df


def _simple_stratified_split(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    """
    Simple stratified split (fallback when MMseqs2 unavailable).
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label"],
        random_state=seed,
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)
