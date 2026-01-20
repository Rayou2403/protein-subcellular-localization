"""
Training loop with early stopping and checkpointing.
"""

import os
import time
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..eval.metrics import compute_metrics


class Trainer:
    """
    Handles training, validation, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        checkpoint_dir: str,
        early_stopping_patience: int = 15,
        gradient_clip_norm: Optional[float] = None,
        use_amp: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_norm = gradient_clip_norm
        self.use_amp = use_amp

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Early stopping state
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            esmc_emb = batch["esmc_embeddings"].to(self.device)
            prostt5_emb = batch["prostt5_embeddings"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(esmc_emb, prostt5_emb)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, labels)
            else:
                outputs = self.model(esmc_emb, prostt5_emb)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(all_labels, all_preds)

        return {"loss": avg_loss, **metrics}

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                esmc_emb = batch["esmc_embeddings"].to(self.device)
                prostt5_emb = batch["prostt5_embeddings"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(esmc_emb, prostt5_emb)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_labels, all_preds)

        return {"loss": avg_loss, **metrics}

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Full training loop with early stopping.

        Returns:
            Best validation metrics
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['macro_f1']:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['macro_f1']:.4f}, "
                  f"MCC: {val_metrics['mcc']:.4f}")

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Checkpointing and early stopping
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0

                # Save best model
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"Saved best model (F1: {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1

            # Save last model
            last_checkpoint_path = os.path.join(self.checkpoint_dir, "last_model.pt")
            self.save_checkpoint(last_checkpoint_path, epoch, val_metrics)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print(f"\nTraining complete. Best Val F1: {self.best_val_f1:.4f}")
        return {"best_val_f1": self.best_val_f1, "best_val_loss": self.best_val_loss}

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint
