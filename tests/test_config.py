import pytest
from pydantic import ValidationError

from moltab.config import Settings


def test_defaults():
    s = Settings(data_path="x.csv", task="regression")
    assert (s.smiles_col, s.target_col, s.n_estimators, s.port, s.kv_cache) == (
        "smiles",
        "target",
        8,
        8000,
        True,
    )


def test_env_vars(monkeypatch):
    monkeypatch.setenv("MOLTAB_DATA_PATH", "/data/t.csv")
    monkeypatch.setenv("MOLTAB_TASK", "classification")
    monkeypatch.setenv("MOLTAB_PORT", "9000")
    s = Settings()
    assert s.task == "classification" and s.port == 9000 and str(s.data_path) == "/data/t.csv"


def test_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("MOLTAB_PORT", "9000")
    assert Settings(data_path="a", task="regression", port=1234).port == 1234


def test_invalid_task():
    with pytest.raises(ValidationError):
        Settings(data_path="a", task="ranking")


def test_missing_required(monkeypatch):
    monkeypatch.delenv("MOLTAB_TASK", raising=False)
    monkeypatch.delenv("MOLTAB_DATA_PATH", raising=False)
    with pytest.raises(ValidationError):
        Settings()
