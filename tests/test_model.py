import numpy as np
import pytest

from moltab.model import ScoringModel


def Xy(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    return X, X[:, 0] * 2


def test_factory_receives_settings(estimator_factory):
    m = ScoringModel(
        "regression",
        n_estimators=3,
        device="cpu",
        kv_cache=False,
        estimator_factory=estimator_factory,
    )
    assert m._est.task == "regression"
    assert m._est.kwargs == {"n_estimators": 3, "device": "cpu", "kv_cache": False}


def test_default_factory_picks_class():
    from tabicl import TabICLClassifier, TabICLRegressor

    assert isinstance(ScoringModel("regression")._est, TabICLRegressor)
    assert isinstance(ScoringModel("classification")._est, TabICLClassifier)


def test_regression_fit_predict(estimator_factory):
    X, y = Xy()
    m = ScoringModel("regression", estimator_factory=estimator_factory).fit(X, y)
    assert np.abs(m.predict(X) - y).max() < 0.2
    assert m.classes is None
    with pytest.raises(RuntimeError, match="classification"):
        m.predict_proba(X)


def test_classification_label_roundtrip(estimator_factory):
    X, y = Xy()
    labels = np.where(y > 0, "pos", "neg")
    m = ScoringModel("classification", estimator_factory=estimator_factory).fit(X, labels)
    assert m.classes == ["neg", "pos"]
    assert set(m.predict(X)) <= {"neg", "pos"}
    proba = m.predict_proba(X)
    assert proba.shape == (20, 2) and np.allclose(proba.sum(1), 1)
    assert (m.predict(X) == labels).mean() > 0.9


def test_predict_before_fit(estimator_factory):
    m = ScoringModel("regression", estimator_factory=estimator_factory)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict(np.zeros((1, 4)))


@pytest.mark.slow
def test_real_tabicl_smoke():
    X, y = Xy(60)
    m = ScoringModel("regression", n_estimators=2, device="cpu").fit(X, y)
    assert m.predict(X[:5]).shape == (5,)
