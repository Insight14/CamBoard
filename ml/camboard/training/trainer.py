"""Generic model Trainer supporting early stopping, best checkpoint retention, and metrics."""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from camboard.training.metrics import (
    accuracy,
    confusion_matrix_fig,
    per_class_recall,
    save_confusion_matrix,
    top_k_accuracy,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Standardized training harness for CamBoard models.

    Key behaviors:
    - Monitors validation loss or accuracy
    - Implements early stopping with configurable patience
    - Retains only the best validation checkpoint (not the last)
    - Supports cosine annealing or OneCycle learning rate schedulers
    - Reports test set performance strictly ONCE at conclusion
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config

        # Training parameters
        self.max_epochs = int(config.get("max_epochs", 50))
        self.patience = int(config.get("early_stop_patience", 10))
        self.monitor = str(config.get("monitor", "val_acc")).lower()
        self.model_name = str(config.get("model_name", "model"))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"

        # Internal state tracking
        self.best_metric: float = -float("inf") if "acc" in self.monitor else float("inf")
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.patience_counter = 0
        self.best_epoch = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Execute full training loop with early stopping."""
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        logger.info(
            f"Starting training '{self.model_name}' on {self.device} for up to {self.max_epochs} epochs (patience={self.patience})."
        )

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._validate_one_epoch(val_loader)

            # Step scheduler if per-epoch
            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Check if this is the best epoch
            is_better = False
            current_metric = val_acc if "acc" in self.monitor else val_loss
            if "acc" in self.monitor:
                if current_metric > self.best_metric:
                    is_better = True
            else:
                if current_metric < self.best_metric:
                    is_better = True

            if is_better:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(epoch, current_metric)
            else:
                self.patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or is_better:
                print(
                    f"Epoch [{epoch:03d}/{self.max_epochs:03d}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% "
                    f"{'(*)' if is_better else ''}"
                )

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {self.best_epoch}.")
                break

        # Always reload best checkpoint
        self._load_best_checkpoint()
        return history

    def _train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            inputs, labels = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Support tuple outputs (e.g. classification + auxiliary orientation head)
            if isinstance(outputs, tuple):
                logits = outputs[0]
                loss = self.criterion(logits, labels)
                if len(outputs) > 1 and len(batch) > 2:
                    # auxiliary loss
                    aux_labels = batch[2].to(self.device)
                    aux_weight = float(self.config.get("orientation_loss_weight", 0.3))
                    loss = loss + aux_weight * nn.functional.cross_entropy(outputs[1], aux_labels)
            else:
                logits = outputs
                loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            total_loss += float(loss.item()) * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += len(labels)

        return total_loss / max(1, total), correct / max(1, total)

    def _validate_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                inputs, labels = self._unpack_batch(batch)
                outputs = self.model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.criterion(logits, labels)

                total_loss += float(loss.item()) * len(labels)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == labels).sum().item())
                total += len(labels)

        return total_loss / max(1, total), correct / max(1, total)

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            return inputs, labels
        raise ValueError("Unexpected batch format.")

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        payload = {
            "epoch": epoch,
            "metric": metric,
            "monitor": self.monitor,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        torch.save(payload, str(self.best_checkpoint_path))

    def _load_best_checkpoint(self) -> None:
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        elif self.best_checkpoint_path.exists():
            ckpt = torch.load(str(self.best_checkpoint_path), map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

    def evaluate(
        self,
        loader: DataLoader,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """Evaluate best model on held-out test data strictly once."""
        self._load_best_checkpoint()
        self.model.eval()

        all_preds = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for batch in loader:
                inputs, labels = self._unpack_batch(batch)
                outputs = self.model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_scores.append(probs)

        all_scores_arr = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0, len(class_names)))
        acc = accuracy(all_labels, all_preds)
        recalls = per_class_recall(all_labels, all_preds, class_names)
        cm_fig = confusion_matrix_fig(
            all_labels,
            all_preds,
            class_names,
            title=f"{self.model_name} Test Confusion Matrix",
        )

        results: Dict[str, Any] = {
            "accuracy": acc,
            "per_class_recall": recalls,
            "confusion_fig": cm_fig,
            "y_true": all_labels,
            "y_pred": all_preds,
        }

        if len(class_names) >= 5 and all_scores_arr.shape[0] > 0:
            results["top5_accuracy"] = top_k_accuracy(np.array(all_labels), all_scores_arr, k=5)

        return results

