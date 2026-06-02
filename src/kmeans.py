from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KMeans:
    n_clusters: int = 3
    max_iter: int = 300
    tol: float = 1e-6
    random_state: Optional[int] = None

    centroids: Optional[np.ndarray] = field(default=None, init=False)
    labels_: Optional[np.ndarray] = field(default=None, init=False)
    n_iter_: int = field(default=0, init=False)
    inertia_: float = field(default=0.0, init=False)
    centroids_history_: list[np.ndarray] = field(default_factory=list, init=False)
    inertia_history_: list[float] = field(default_factory=list, init=False)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Randomly pick K rows of X as initial centroids."""
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters must be <= number of samples")

        rng = self._rng()
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].astype(float).copy()

    @staticmethod
    def _squared_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return pairwise squared Euclidean distances between X and centroids."""
        # Shape: (n_samples, 1, n_features) - (1, K, n_features)
        diff = X[:, None, :] - centroids[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """E-step: assign each point to its closest centroid."""
        distances = self._squared_distances(X, centroids)
        return np.argmin(distances, axis=1)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        old_centroids: np.ndarray,
    ) -> np.ndarray:
        """M-step: update centroids by averaging assigned points.

        If a cluster becomes empty, keep its previous centroid. This prevents NaN.
        """
        new_centroids = old_centroids.copy()

        for k in range(self.n_clusters):
            Xk = X[labels == k]
            if len(Xk) > 0:
                new_centroids[k] = np.mean(Xk, axis=0)

        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Sum of squared distances from each point to its assigned centroid."""
        return float(np.sum((X - centroids[labels]) ** 2))

    def _has_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """Return True if the centroids do not move significantly."""
        centroid_shift = np.linalg.norm(new_centroids - old_centroids)
        return centroid_shift <= self.tol

    def fit(self, X: np.ndarray) -> "KMeans":
        """Train K-Means with EM iterations."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features)")

        current_centroids = self._init_centroids(X)
        self.centroids_history_ = [current_centroids.copy()]
        self.inertia_history_ = []

        for iteration in range(1, self.max_iter + 1):
            # E-step
            labels = self._assign_labels(X, current_centroids)

            # M-step
            new_centroids = self._update_centroids(X, labels, current_centroids)
            inertia = self._compute_inertia(X, labels, new_centroids)

            self.centroids_history_.append(new_centroids.copy())
            self.inertia_history_.append(inertia)

            if self._has_converged(current_centroids, new_centroids):
                current_centroids = new_centroids
                self.n_iter_ = iteration
                break

            # Important: update current centroids before the next loop.
            current_centroids = new_centroids
        else:
            self.n_iter_ = self.max_iter

        self.centroids = current_centroids
        self.labels_ = self._assign_labels(X, self.centroids)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels."""
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest learned centroid."""
        if self.centroids is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.centroids)
