from __future__ import annotations

from itertools import permutations

import numpy as np


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int) -> float:
    """Compute clustering accuracy after best label permutation.

    K-Means labels are arbitrary. For example, predicted label 0 may correspond
    to true class 2. This function tries all permutations, which is fine for K=3.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    best = 0
    for perm in permutations(range(n_clusters)):
        mapped = np.array([perm[label] for label in y_pred])
        correct = int(np.sum(mapped == y_true))
        best = max(best, correct)

    return best / len(y_true)


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return a K x K count table: rows=true labels, columns=predicted labels."""
    table = np.zeros((n_clusters, n_clusters), dtype=int)
    for t, p in zip(y_true, y_pred):
        table[int(t), int(p)] += 1
    return table
