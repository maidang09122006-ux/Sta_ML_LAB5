from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot predicted clusters and final centroids."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=18, alpha=0.75)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.2,
        label="Centroids",
    )
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_inertia(history: list[float], title: str, save_path: str | Path) -> None:
    """Plot inertia decreasing over EM iterations."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Inertia / Within-cluster SSE")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
