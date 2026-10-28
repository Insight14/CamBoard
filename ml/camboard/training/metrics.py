"""Evaluation metrics, recall tables, and confusion matrix visualizations."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


def accuracy(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Calculate standard multi-class classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def top_k_accuracy(y_true: np.ndarray, y_scores: np.ndarray, k: int = 5) -> float:
    """Calculate top-k categorical accuracy."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if y_scores.shape[1] < k:
        k = y_scores.shape[1]
    top_k_indices = np.argsort(y_scores, axis=1)[:, -k:]
    correct = np.any(top_k_indices == y_true[:, None], axis=1)
    return float(np.mean(correct))


def per_class_recall(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: List[str],
) -> Dict[str, float]:
    """Compute recall for each class name."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    recalls = {}
    for idx, name in enumerate(class_names):
        mask = (y_true_arr == idx)
        if np.sum(mask) == 0:
            recalls[name] = 1.0  # vacuously perfect if no true instances
        else:
            recalls[name] = float(np.mean(y_pred_arr[mask] == idx))
    return recalls


def confusion_matrix_fig(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Create a row-normalized (recall-style) confusion matrix seaborn heatmap figure."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Normalize by row (true labels) with epsilon safeguard
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig_size = max(6, min(14, len(class_names) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    show_annot = len(class_names) <= 28
    sns.heatmap(
        cm_norm,
        annot=show_annot,
        fmt=".2f" if show_annot else "",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    return fig


def save_confusion_matrix(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Save matplotlib figure to disk and close it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)

