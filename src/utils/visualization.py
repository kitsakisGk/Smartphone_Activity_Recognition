"""Visualization utilities for training metrics and results."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def plot_training_history(
    history: Dict,
    metrics: List[str],
    save_path: Optional[Path] = None,
    title: str = "Training History"
) -> None:
    """
    Plot training history for multiple metrics.

    Args:
        history: Training history dictionary from model.fit()
        metrics: List of metrics to plot
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f"Train {metric}", linewidth=2)

            val_metric = f"val_{metric}"
            if val_metric in history:
                ax.plot(history[val_metric], label=f"Val {metric}", linewidth=2)

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f"{metric.capitalize()} over Epochs", fontsize=13, fontweight="bold")
            ax.legend(framealpha=0.9)
            ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_metrics_plot(
    history: Dict,
    model_name: str,
    save_dir: Path,
    plot_type: str = "accuracy"
) -> Path:
    """
    Save specific metric plots for a model.

    Args:
        history: Training history dictionary
        model_name: Name of the model (e.g., 'CNN', 'CNN_LSTM')
        save_dir: Directory to save plots
        plot_type: Type of plot - 'accuracy' or 'metrics'

    Returns:
        Path to saved plot
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "accuracy":
        # Plot accuracy and loss
        ax.plot(history["accuracy"], label="Train Accuracy", linewidth=2)
        ax.plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
        ax.plot(history["loss"], label="Train Loss", linewidth=2, linestyle="--")
        ax.plot(history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
        ax.set_title(f"{model_name} - Accuracy and Loss", fontsize=14, fontweight="bold")
        ax.set_ylabel("Value")
        save_name = f"Model_Accuracy_{model_name}.png"

    else:  # metrics
        # Plot F1, Precision, Recall
        ax.plot(history["f1_score"], label="Train F1-Score", linewidth=2)
        ax.plot(history["val_f1_score"], label="Val F1-Score", linewidth=2)
        ax.plot(history["precision"], label="Train Precision", linewidth=2)
        ax.plot(history["val_precision"], label="Val Precision", linewidth=2)
        ax.plot(history["recall"], label="Train Recall", linewidth=2)
        ax.plot(history["val_recall"], label="Val Recall", linewidth=2)
        ax.set_title(f"{model_name} - Performance Metrics", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score")
        save_name = f"Model_Metrics_{model_name}.png"

    ax.set_xlabel("Epoch")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    save_path = save_dir / save_name
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path
