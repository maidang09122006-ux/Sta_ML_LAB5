from __future__ import annotations

from pprint import pprint

from src.experiments import run_all


if __name__ == "__main__":
    results = run_all(output_dir="outputs")
    print("Done. Results were saved to outputs/summary.csv")
    pprint(results)
