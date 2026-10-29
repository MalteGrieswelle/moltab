"""5x5 repeated CV of the moltab regression pipeline on a CSV; prints metrics and saves a boxplot.

uv run python scripts/cross_validate.py data/train.csv --smiles-col smiles --target-col target
"""

import argparse
from pathlib import Path

from moltab.data import load_training_data
from moltab.evaluate import boxplot, cross_validate
from moltab.featurizer import CheMeleonFeaturizer
from moltab.model import ScoringModel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-col", default="target")
    p.add_argument("--n-estimators", type=int, default=8)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="plot path (default: <csv>_cv.png)")
    a = p.parse_args()

    train = load_training_data(a.csv, a.smiles_col, a.target_col, "regression")
    X = CheMeleonFeaturizer(a.device).transform(train.smiles)
    res = cross_validate(
        X, train.y, lambda: ScoringModel("regression", a.n_estimators, a.device), seed=a.seed
    )
    print(f"{len(train.smiles)} molecules, {len(res)} folds")
    print(res.agg(["mean", "std", "median"]).round(3).to_string())
    out = a.out or a.csv.with_name(a.csv.stem + "_cv.png")
    boxplot(res, out, f"{a.csv.name}: 5x5 CV (n={len(train.smiles)})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
