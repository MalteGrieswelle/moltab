import numpy as np
import pytest

from moltab.evaluate import METRICS, boxplot, cross_validate, regression_metrics
from moltab.model import ScoringModel


def test_metrics_perfect_and_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m == pytest.approx({"R2": 1.0, "MAE": 0.0, "MSE": 0.0, "Kendall tau": 1.0})
    m = regression_metrics(y, y[::-1])
    assert m["Kendall tau"] == -1.0 and m["MAE"] == 2.0 and m["MSE"] == 5.0


def test_cross_validate_shape_and_fresh_models(estimator_factory):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    y = X[:, 0] * 2
    made = []

    def factory():
        made.append(ScoringModel("regression", estimator_factory=estimator_factory))
        return made[-1]

    res = cross_validate(X, y, factory, n_splits=5, n_repeats=5)
    assert res.shape == (25, 4) and list(res.columns) == METRICS
    assert len(made) == 25 and all(m._est.fit_calls == 1 for m in made)
    assert res["R2"].mean() > 0.9


def test_cross_validate_deterministic(estimator_factory):
    X = np.random.default_rng(1).normal(size=(30, 3))
    y = X[:, 0]
    f = lambda: ScoringModel("regression", estimator_factory=estimator_factory)
    a = cross_validate(X, y, f, seed=3)
    b = cross_validate(X, y, f, seed=3)
    assert a.equals(b)


def test_boxplot_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    import pandas as pd

    res = pd.DataFrame(np.random.default_rng(0).normal(size=(25, 4)), columns=METRICS)
    out = tmp_path / "p.png"
    boxplot(res, out)
    assert out.stat().st_size > 1000
