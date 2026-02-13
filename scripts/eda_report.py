"""
Generate an EDA report with figures for the DeepLocPro splits file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_SPLITS = "data/processed/splits.csv"
DEFAULT_OUT = "report"
DEFAULT_SAMPLE_FRAC = None
DEFAULT_SAMPLE_N = None
LABEL_MAP = {
    "Cytoplasmic": "Cytoplasmic",
    "CytoplasmicMembrane": "Cytoplasmic Membrane",
    "CYtoplasmicMembrane": "Cytoplasmic Membrane",
    "Extracellular": "Extracellular",
    "OuterMembrane": "Outer Membrane",
    "Periplasmic": "Periplasmic",
    "Cellwall": "Cell Wall",
    "CellWall": "Cell Wall",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _add_seq_len(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["seq_len"] = df["sequence"].astype(str).str.len()
    if "label" in df.columns:
        df["label_display"] = df["label"].map(LABEL_MAP).fillna(df["label"])
    return df


def _sample_df(df: pd.DataFrame, frac: float | None, n: int | None, seed: int) -> pd.DataFrame:
    if frac is not None:
        if not (0.0 < frac <= 1.0):
            raise ValueError("--sample_frac must be in (0, 1].")
        return df.sample(frac=frac, random_state=seed)
    if n is not None:
        if n < 1:
            raise ValueError("--sample_n must be >= 1.")
        return df.sample(n=min(n, len(df)), random_state=seed)
    return df


def _make_overview(fig_dir: Path, out_path: Path) -> None:
    images = sorted(fig_dir.glob("*.png"))
    if not images:
        return
    cols = 2
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(10, 4 * rows))
    for idx, img_path in enumerate(images, start=1):
        ax = plt.subplot(rows, cols, idx)
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(img_path.stem.replace("_", " "))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures and summary.")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="Path to splits CSV")
    parser.add_argument("--out_dir", default=DEFAULT_OUT, help="Output directory (report)")
    parser.add_argument("--sample_frac", type=float, default=DEFAULT_SAMPLE_FRAC, help="Sample fraction for plots")
    parser.add_argument("--sample_n", type=int, default=DEFAULT_SAMPLE_N, help="Sample size for plots")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)

    df = pd.read_csv(args.splits)
    df = _add_seq_len(df)
    plot_df = _sample_df(df, args.sample_frac, args.sample_n, args.seed)

    summary = []
    summary.append("# EDA Summary\n")
    summary.append(f"Rows: {len(df)}")
    summary.append(f"Columns: {', '.join(df.columns)}")
    summary.append("")

    missing = df.isna().mean().sort_values(ascending=False)
    summary.append("## Missingness")
    summary.append(missing.to_string())
    summary.append("")

    label_counts = df["label_display"].value_counts()
    summary.append("## Label distribution")
    summary.append(label_counts.to_string())
    summary.append("")

    seq_stats = df["seq_len"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    summary.append("## Sequence length stats")
    summary.append(seq_stats.to_string())
    summary.append("")

    p10 = int(seq_stats["10%"])
    p25 = int(seq_stats["25%"])
    p90 = int(seq_stats["90%"])
    p95 = int(seq_stats["95%"])
    p99 = int(seq_stats["99%"])
    summary.append("## Suggested max_len for quick tests")
    summary.append(f"- max_len {p10}: keeps ~10% of sequences (very small quick test)")
    summary.append(f"- max_len {p25}: keeps ~25% of sequences (small quick test)")
    summary.append(f"- max_len 150: very fast, may drop longer sequences")
    summary.append(f"- max_len 200: good quick test")
    summary.append(f"- max_len {p90}: keeps ~90% of sequences")
    summary.append(f"- max_len {p95}: keeps ~95% of sequences")
    summary.append(f"- max_len {p99}: keeps ~99% of sequences")
    summary.append("")

    if "gram_type" in df.columns:
        gram_counts = df["gram_type"].value_counts()
    summary.append("## Gram type distribution")
    summary.append(gram_counts.to_string())
    summary.append("")

    if "split" in df.columns:
        split_counts = df["split"].value_counts()
    summary.append("## Split distribution")
    summary.append(split_counts.to_string())
    summary.append("")

    summary.append("## Conclusion (FR)")
    summary.append(
        "Les distributions par classe et par split montrent les volumes disponibles pour "
        "l'apprentissage. La longueur des sequences varie fortement, ce qui augmente le cout "
        "de calcul pendant l'extraction d'embeddings. En pratique, `max_len=1000` est un bon "
        "compromis CPU: couverture elevee du dataset tout en restant faisable avec ProstT5 3Di."
    )
    summary.append(
        "Exemple: `python -m src.embeddings.fetch_embeddings --esm_fasta data/raw/graphpart_set.fasta "
        "--esm_out data/processed/embeddings/esmc.h5 --prost_out data/processed/embeddings/prostt5.h5 "
        "--embed2_backend prostt5 --esm_batch 16 --prost_batch 1 --max_len 1000 "
        "--prost_offload_dir data/interim/offload --prost_max_memory 6GB`."
    )

    # Label distribution bar plot
    plt.figure(figsize=(8, 4))
    sns.countplot(data=plot_df, x="label_display", order=label_counts.index)
    plt.xticks(rotation=30, ha="right")
    plt.title("Label distribution")
    _save_fig(fig_dir / "label_distribution.png")

    # Sequence length histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(plot_df["seq_len"], bins=50)
    plt.title("Sequence length distribution")
    plt.xlabel("Sequence length")
    _save_fig(fig_dir / "sequence_length_hist.png")

    # Sequence length by label (boxplot)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="label_display", y="seq_len")
    plt.xticks(rotation=30, ha="right")
    plt.title("Sequence length by label")
    _save_fig(fig_dir / "sequence_length_boxplot.png")

    # Gram type distribution
    if "gram_type" in df.columns:
        plt.figure(figsize=(4, 4))
        sns.countplot(data=plot_df, x="gram_type")
        plt.title("Gram type distribution")
        _save_fig(fig_dir / "gram_type_distribution.png")

    # Split distribution
    if "split" in df.columns:
        plt.figure(figsize=(4, 4))
        sns.countplot(data=plot_df, x="split")
        plt.title("Split distribution")
        _save_fig(fig_dir / "split_distribution.png")

    _make_overview(fig_dir, fig_dir / "overview.png")

    (out_dir / "eda.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved figures to {fig_dir}")
    print(f"Saved summary to {out_dir / 'eda.md'}")


if __name__ == "__main__":
    main()
