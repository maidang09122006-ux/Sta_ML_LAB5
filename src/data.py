from __future__ import annotations

from typing import Iterable

import numpy as np


IDENTITY_COV = np.array([[1.0, 0.0], [0.0, 1.0]])


def make_gaussian_dataset(
    means: Iterable[Iterable[float]],
    covariances: Iterable[np.ndarray],
    sizes: Iterable[int],
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data from multiple 2D Gaussian distributions.

    Returns
    -------
    X:
        Data matrix with shape (n_samples, 2).
    y:
        Ground-truth component labels. These labels are only for evaluation/plotting,
        not used by K-Means during training.
    """
    rng = np.random.default_rng(random_state)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for label, (mean, cov, size) in enumerate(zip(means, covariances, sizes)):
        Xi = rng.multivariate_normal(mean=np.asarray(mean), cov=np.asarray(cov), size=size)
        yi = np.full(size, label, dtype=int)
        X_parts.append(Xi)
        y_parts.append(yi)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    if shuffle:
        order = rng.permutation(len(X))
        X = X[order]
        y = y[order]

    return X, y


def assignment1_data(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Assignment 1: balanced clusters with identical covariance."""
    means = [(2, 2), (8, 3), (3, 6)]
    covariances = [IDENTITY_COV, IDENTITY_COV, IDENTITY_COV]
    sizes = [200, 200, 200]
    return make_gaussian_dataset(means, covariances, sizes, random_state=random_state)


def assignment2_data(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Assignment 2: imbalanced clusters.

    The notebook states cluster sizes 1200, 200, and 1000, so the total is 2400.
    """
    means = [(2, 2), (8, 3), (3, 6)]
    covariances = [IDENTITY_COV, IDENTITY_COV, IDENTITY_COV]
    sizes = [1200, 200, 1000]
    return make_gaussian_dataset(means, covariances, sizes, random_state=random_state)


def assignment3_data(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Assignment 3: one elongated Gaussian distribution."""
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma2 = np.array([[10.0, 0.0], [0.0, 1.0]])
    means = [(2, 2), (8, 3), (3, 6)]
    covariances = [sigma1, sigma1, sigma2]
    sizes = [200, 200, 200]
    return make_gaussian_dataset(means, covariances, sizes, random_state=random_state)
