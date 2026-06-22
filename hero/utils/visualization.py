from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_training_metrics_grid(history: list[dict], output_path: str | Path) -> None:
    if not history:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    series = [
        ("loss", "Loss"),
        ("dice", "Dice"),
        ("t_acc", "T-stage Accuracy"),
        ("n_acc", "N-stage Accuracy"),
        ("relapse_acc", "Relapse Accuracy"),
        ("cindex", "C-index"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, series):
        values = [item.get(key, 0.0) for item in history]
        ax.plot(epochs, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
