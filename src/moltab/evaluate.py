"""Repeated k-fold cross-validation of the regression pipeline."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from moltab.model import ScoringModel

METRICS = ["R2", "MAE", "MSE", "Kendall tau"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "Kendall tau": kendalltau(y_true, y_pred).statistic,
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], ScoringModel],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per fold (n_splits * n_repeats), columns = METRICS; a fresh model is fit per fold."""
    rows = []
    for train, test in RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X):
        model = model_factory().fit(X[train], y[train])
        rows.append(regression_metrics(y[test], model.predict(X[test])))
    return pd.DataFrame(rows, columns=METRICS)


def boxplot(results: pd.DataFrame, path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3 * len(METRICS), 4))
    for ax, name in zip(axes, METRICS):
        ax.boxplot(results[name].dropna(), widths=0.5)
        ax.set_xticks([])
        ax.set_title(name)
    fig.suptitle(title or f"{len(results)} CV folds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
