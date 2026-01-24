#!/usr/bin/env python3
"""Generate evaluation figures from results/evaluation JSON files."""
import json
import os

import matplotlib.pyplot as plt


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_metrics(metrics: dict, out_path: str) -> None:
    labels = ["Accuracy", "Macro F1", "MCC"]
    values = [metrics["accuracy"], metrics["macro_f1"], metrics["mcc"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, values, color=["#2563eb", "#16a34a", "#9333ea"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Global evaluation metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_class(per_class: dict, out_path: str) -> None:
    classes = list(per_class.keys())
    f1 = [per_class[c]["f1"] for c in classes]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = list(range(len(classes)))
    bars = ax.bar(x, f1, color="#0ea5e9")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 scores")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")

    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    metrics_path = os.path.join("results", "evaluation", "metrics.json")
    per_class_path = os.path.join("results", "evaluation", "per_class_metrics.json")
    out_dir = os.path.join("reports", "figures")
    os.makedirs(out_dir, exist_ok=True)

    metrics = _load_json(metrics_path)
    per_class = _load_json(per_class_path)

    plot_metrics(metrics, os.path.join(out_dir, "results_metrics.png"))
    plot_per_class(per_class, os.path.join(out_dir, "results_per_class_f1.png"))


if __name__ == "__main__":
    main()
