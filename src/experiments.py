from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from .data import assignment1_data, assignment2_data, assignment3_data
from .kmeans import KMeans
from .metrics import clustering_accuracy, confusion_table
from .plotting import plot_clusters, plot_inertia


DatasetFactory = Callable[[int], tuple[np.ndarray, np.ndarray]]


def run_one_experiment(
    name: str,
    dataset_factory: DatasetFactory,
    data_seed: int,
    init_seed: int,
    output_dir: str | Path,
) -> dict[str, object]:
    """Generate data, train K-Means, save plots, and return metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y_true = dataset_factory(data_seed)
    model = KMeans(n_clusters=3, max_iter=300, tol=1e-6, random_state=init_seed)
    y_pred = model.fit_predict(X)

    acc = clustering_accuracy(y_true, y_pred, n_clusters=3)
    table = confusion_table(y_true, y_pred, n_clusters=3)

    safe_name = name.lower().replace(" ", "_")
    plot_clusters(
        X,
        y_pred,
        model.centroids,
        title=f"{name} - K-Means result - init seed {init_seed}",
        save_path=output_dir / f"{safe_name}_seed_{init_seed}_clusters.png",
    )
    plot_inertia(
        model.inertia_history_,
        title=f"{name} - Inertia over iterations - init seed {init_seed}",
        save_path=output_dir / f"{safe_name}_seed_{init_seed}_inertia.png",
    )

    return {
        "assignment": name,
        "data_seed": data_seed,
        "init_seed": init_seed,
        "n_samples": len(X),
        "iterations": model.n_iter_,
        "inertia": round(model.inertia_, 4),
        "accuracy_after_best_label_mapping": round(acc, 4),
        "centroids": np.round(model.centroids, 4).tolist(),
        "confusion_table": table.tolist(),
    }


def run_all(output_dir: str | Path = "outputs") -> list[dict[str, object]]:
    """Run all K-Means assignments and save a CSV summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    # Assignment 1: run multiple initializations to discuss random centroid effect.
    for init_seed in [0, 1, 2, 3, 4]:
        results.append(
            run_one_experiment(
                name="Assignment 1",
                dataset_factory=assignment1_data,
                data_seed=42,
                init_seed=init_seed,
                output_dir=output_dir,
            )
        )

    # Assignment 2 and 3: one representative run is enough for the required comment.
    results.append(
        run_one_experiment(
            name="Assignment 2",
            dataset_factory=assignment2_data,
            data_seed=42,
            init_seed=0,
            output_dir=output_dir,
        )
    )
    results.append(
        run_one_experiment(
            name="Assignment 3",
            dataset_factory=assignment3_data,
            data_seed=42,
            init_seed=0,
            output_dir=output_dir,
        )
    )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "assignment",
                "data_seed",
                "init_seed",
                "n_samples",
                "iterations",
                "inertia",
                "accuracy_after_best_label_mapping",
                "centroids",
                "confusion_table",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return results
