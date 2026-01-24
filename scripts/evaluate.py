"""
Evaluation script for trained model.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import ProteinLocalizationDataset, collate_variable_length
from src.models import DualEmbeddingFusionModel, TransformerFusionModel
from src.eval import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (test/val)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_test_loader(config: dict, split: str = "test"):
    """Create test data loader."""
    data_config = config["data"]
    training_config = config["training"]

    test_dataset = ProteinLocalizationDataset(
        metadata_path=data_config["metadata_path"],
        esmc_h5_path=data_config["esmc_h5_path"],
        prostt5_h5_path=data_config["prostt5_h5_path"],
        split=split,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
    )

    return test_loader, test_dataset


def create_model(config: dict) -> torch.nn.Module:
    """Create model from config."""
    model_config = config["model"]
    model_type = model_config.get("type", "gated_mlp")

    if model_type == "transformer_mlp":
        return TransformerFusionModel(
            esmc_dim=model_config["esmc_dim"],
            prostt5_dim=model_config["prostt5_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_attention_heads=model_config.get("num_attention_heads", 4),
            num_transformer_layers=model_config.get("num_transformer_layers", 2),
            dropout=model_config["dropout"],
            num_classes=model_config["num_classes"],
        )

    if model_type == "gated_mlp":
        return DualEmbeddingFusionModel(
            esmc_dim=model_config["esmc_dim"],
            prostt5_dim=model_config["prostt5_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_attention_heads=model_config.get("num_attention_heads", 8),
            num_fusion_layers=model_config.get("num_fusion_layers", 2),
            dropout=model_config["dropout"],
            num_classes=model_config["num_classes"],
            pooling_type=model_config.get("pooling_type", "attention"),
            use_gated_fusion=model_config.get("use_gated_fusion", True),
        )

    raise ValueError(f"Unknown model.type: {model_type}")


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    model = create_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    if args.output_dir is None:
        output_dir = config["paths"]["output_dir"]
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create test loader
    print(f"\nLoading {args.split} data...")
    test_loader, test_dataset = create_test_loader(config, split=args.split)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, config, device)

    # Get class names
    class_names = config.get("class_names", None)
    if class_names is None:
        class_names = [test_dataset.idx_to_label[i] for i in range(test_dataset.num_classes)]

    # Evaluate
    print("\n" + "=" * 80)
    print("Starting evaluation")
    print("=" * 80)

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
