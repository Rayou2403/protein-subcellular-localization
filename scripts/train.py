"""
Training script for dual-embedding protein localization.
"""

import argparse
import os
import sys
import random
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import ProteinLocalizationDataset, collate_variable_length
from src.models import DualEmbeddingFusionModel, TransformerFusionModel
from src.training import Trainer, get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(description="Train protein localization model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_data_loaders(config: dict):
    """Create train/val data loaders."""
    data_config = config["data"]
    training_config = config["training"]

    # Create datasets
    train_dataset = ProteinLocalizationDataset(
        metadata_path=data_config["metadata_path"],
        esmc_h5_path=data_config["esmc_h5_path"],
        prostt5_h5_path=data_config["prostt5_h5_path"],
        split="train",
    )

    val_dataset = ProteinLocalizationDataset(
        metadata_path=data_config["metadata_path"],
        esmc_h5_path=data_config["esmc_h5_path"],
        prostt5_h5_path=data_config["prostt5_h5_path"],
        split="val",
        label_to_idx=train_dataset.label_to_idx,
    )

    # Balanced sampler for training
    if training_config.get("use_balanced_sampler", False):
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_variable_length,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


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


def create_optimizer(model: torch.nn.Module, config: dict):
    """Create optimizer from config."""
    training_config = config["training"]
    optimizer_config = config["optimizer"]

    if optimizer_config["type"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            betas=optimizer_config["betas"],
            eps=optimizer_config["eps"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")

    return optimizer


def create_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler from config."""
    scheduler_config = config["scheduler"]
    training_config = config["training"]

    if scheduler_config["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config["num_epochs"] - scheduler_config["warmup_epochs"],
            eta_min=scheduler_config["min_lr"],
        )
    elif scheduler_config["type"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
    elif scheduler_config["type"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=scheduler_config["min_lr"],
        )
    else:
        scheduler = None

    return scheduler


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Experiment: {config['experiment_name']}")

    # Set seed
    set_seed(config["seed"])
    print(f"Set random seed: {config['seed']}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, train_dataset = create_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Num classes: {train_dataset.num_classes}")

    # Get class weights
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"\nClass weights: {class_weights}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create loss function
    training_config = config["training"]
    criterion = get_loss_function(
        loss_type=training_config["loss_type"],
        class_weights=class_weights if training_config["loss_type"] in ["weighted_ce", "focal"] else None,
        focal_gamma=training_config.get("focal_gamma", 2.0),
    )
    print(f"Loss function: {training_config['loss_type']}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    # Create trainer
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=training_config["early_stopping_patience"],
        gradient_clip_norm=training_config.get("gradient_clip_norm"),
        use_amp=training_config.get("use_amp", False),
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    best_metrics = trainer.train(num_epochs=training_config["num_epochs"])

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation F1: {best_metrics['best_val_f1']:.4f}")
    print(f"Best validation loss: {best_metrics['best_val_loss']:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
