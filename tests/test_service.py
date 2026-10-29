import threading
import time

import pytest
from conftest import make_settings

from moltab.data import load_training_data
from moltab.model import ScoringModel
from moltab.service import Scorer, build_scorer


def scorer(featurizer, estimator_factory, csv, task, **kw):
    train = load_training_data(csv, "smiles", "y", task)
    model = ScoringModel(task, estimator_factory=estimator_factory)
    return Scorer(featurizer, model, train, **kw)


def test_regression_order_and_invalid(featurizer, estimator_factory, reg_csv):
    sc = scorer(featurizer, estimator_factory, reg_csv, "regression")
    out = sc.score(["CCO", "bad", "c1ccccc1"])
    assert [o.smiles for o in out] == ["CCO", "bad", "c1ccccc1"]
    assert [o.valid for o in out] == [True, False, True]
    assert out[1].score is None
    assert abs(out[0].score - 1.0) < 0.5 and abs(out[2].score - 5.0) < 0.5


def test_all_invalid(featurizer, estimator_factory, reg_csv):
    out = scorer(featurizer, estimator_factory, reg_csv, "regression").score(["x", "y"])
    assert [o.valid for o in out] == [False, False]


def test_fit_once(featurizer, estimator_factory, reg_csv):
    sc = scorer(featurizer, estimator_factory, reg_csv, "regression")
    sc.score(["CCO"])
    sc.score(["CCN"])
    assert sc.model._est.fit_calls == 1


def test_binary_classification_default_positive(featurizer, estimator_factory, clf_csv):
    sc = scorer(featurizer, estimator_factory, clf_csv, "classification")
    assert sc.positive_class == "other"  # last sorted class
    o = sc.score(["c1ccccc1"])[0]
    assert o.label == "other" and o.score == o.probabilities["other"] > 0.5
    assert pytest.approx(sum(o.probabilities.values())) == 1


def test_explicit_positive_class(featurizer, estimator_factory, clf_csv):
    sc = scorer(featurizer, estimator_factory, clf_csv, "classification", positive_class="alcohol")
    o = sc.score(["CCO"])[0]
    assert o.score == o.probabilities["alcohol"]


def test_unknown_positive_class(featurizer, estimator_factory, clf_csv):
    with pytest.raises(ValueError, match="positive_class"):
        scorer(featurizer, estimator_factory, clf_csv, "classification", positive_class="zzz")


def test_multiclass_uses_top_probability(featurizer, estimator_factory, tmp_path):
    import pandas as pd

    p = tmp_path / "m.csv"
    smi = ["CCO", "CCCO", "c1ccccc1", "c1ccccc1C", "CCN", "CCCN"]
    pd.DataFrame({"smiles": smi, "y": ["o", "o", "ar", "ar", "n", "n"]}).to_csv(p, index=False)
    sc = scorer(featurizer, estimator_factory, p, "classification")
    assert sc.positive_class is None
    o = sc.score(["CCO"])[0]
    assert o.label == "o" and o.score == max(o.probabilities.values())


def test_concurrent_calls_serialized(featurizer, estimator_factory, reg_csv):
    sc = scorer(featurizer, estimator_factory, reg_csv, "regression")
    active, overlap = [0], [False]
    orig = sc.model.predict

    def slow(X):
        active[0] += 1
        overlap[0] |= active[0] > 1

        time.sleep(0.02)
        active[0] -= 1
        return orig(X)

    sc.model.predict = slow
    ts = [threading.Thread(target=sc.score, args=(["CCO"],)) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not overlap[0]


def test_build_scorer(featurizer, estimator_factory, reg_csv):
    s = make_settings(reg_csv, "regression")
    model = ScoringModel("regression", estimator_factory=estimator_factory)
    sc = build_scorer(s, featurizer=featurizer, model=model)
    assert sc.n_train == 10 and sc.device == "cpu"
