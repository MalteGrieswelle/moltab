import pytest
from conftest import make_settings
from fastapi.testclient import TestClient

from moltab.api import create_app
from moltab.data import DataError
from moltab.model import ScoringModel
from moltab.service import build_scorer


def app_for(featurizer, estimator_factory, csv, task, **kw):
    s = make_settings(csv, task, **kw)

    def factory(settings):
        return build_scorer(
            settings, featurizer, ScoringModel(task, estimator_factory=estimator_factory)
        )

    return create_app(s, factory)


def test_health_and_info(featurizer, estimator_factory, reg_csv):
    with TestClient(app_for(featurizer, estimator_factory, reg_csv, "regression")) as c:
        assert c.get("/health").json() == {"status": "ok"}
        info = c.get("/info").json()
        assert info["task"] == "regression" and info["n_train"] == 10 and info["n_features"] == 128


def test_score_single_and_list(featurizer, estimator_factory, reg_csv):
    with TestClient(app_for(featurizer, estimator_factory, reg_csv, "regression")) as c:
        r = c.post("/score", json={"smiles": "CCO"}).json()["results"]
        assert len(r) == 1 and r[0]["valid"]
        r = c.post("/score", json={"smiles": ["CCO", "bad"]}).json()["results"]
        assert [x["valid"] for x in r] == [True, False] and r[1]["score"] is None


def test_classification_response(featurizer, estimator_factory, clf_csv):
    with TestClient(app_for(featurizer, estimator_factory, clf_csv, "classification")) as c:
        assert c.get("/info").json()["classes"] == ["alcohol", "other"]
        r = c.post("/score", json={"smiles": ["CCO"]}).json()["results"][0]
        assert set(r["probabilities"]) == {"alcohol", "other"} and r["label"] == "alcohol"


def test_validation_errors(featurizer, estimator_factory, reg_csv):
    with TestClient(
        app_for(featurizer, estimator_factory, reg_csv, "regression", max_request_size=2)
    ) as c:
        assert c.post("/score", json={"smiles": []}).status_code == 422
        assert c.post("/score", json={}).status_code == 422
        assert c.post("/score", json={"smiles": ["C", "CC", "CCC"]}).status_code == 422
        assert c.post("/score", json={"smiles": ["C", "CC"]}).status_code == 200


def test_scorer_built_once(featurizer, estimator_factory, reg_csv):
    app = app_for(featurizer, estimator_factory, reg_csv, "regression")
    with TestClient(app) as c:
        for _ in range(3):
            c.post("/score", json={"smiles": "CCO"})
    assert app.state.scorer.model._est.fit_calls == 1


def test_startup_fails_on_bad_data(featurizer, estimator_factory, tmp_path):
    app = app_for(featurizer, estimator_factory, tmp_path / "missing.csv", "regression")
    with pytest.raises(DataError), TestClient(app):
        pass
