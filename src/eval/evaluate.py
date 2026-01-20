"""
Model evaluation on test set.
"""

import os
import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .metrics import (
    compute_metrics,
    compute_per_class_metrics,
    compute_confusion_matrix,
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Evaluate model on test set and generate reports.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        class_names: List of class names for visualization
        output_dir: Directory to save outputs

    Returns:
        Dictionary with all evaluation results
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_protein_ids = []
    all_logits = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            esmc_emb = batch["esmc_embeddings"].to(device)
            prostt5_emb = batch["prostt5_embeddings"].to(device)
            labels = batch["labels"]
            protein_ids = batch["protein_ids"]

            outputs = model(esmc_emb, prostt5_emb)
            logits = outputs["logits"]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_protein_ids.extend(protein_ids)
            all_logits.extend(logits.cpu().numpy())

    # Compute metrics
    overall_metrics = compute_metrics(all_labels, all_preds)
    per_class_metrics = compute_per_class_metrics(all_labels, all_preds, class_names)
    conf_matrix = compute_confusion_matrix(all_labels, all_preds)

    print("\nOverall Metrics:")
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {overall_metrics['macro_f1']:.4f}")
    print(f"  MCC: {overall_metrics['mcc']:.4f}")

    print("\nPer-class Metrics:")
    for cls_name, metrics in per_class_metrics.items():
        print(f"  {cls_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Support: {metrics['support']}")

    results = {
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix.tolist(),
    }

    # Save outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(overall_metrics, f, indent=2)

        with open(os.path.join(output_dir, "per_class_metrics.json"), "w") as f:
            json.dump(per_class_metrics, f, indent=2)

        # Save predictions
        predictions_df = pd.DataFrame({
            "protein_id": all_protein_ids,
            "true_label": all_labels,
            "predicted_label": all_preds,
        })
        predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        # Plot confusion matrix
        plot_confusion_matrix(
            conf_matrix,
            class_names=class_names,
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

        print(f"\nResults saved to {output_dir}")

    return results


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix as heatmap.

    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(conf_matrix))]

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()
