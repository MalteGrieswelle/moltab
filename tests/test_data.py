import pandas as pd
import pytest

from moltab.data import DataError, canonicalize, load_training_data


def write(tmp_path, rows, cols=("smiles", "y")):
    p = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=list(cols)).to_csv(p, index=False)
    return p


def test_canonicalize():
    assert canonicalize("OCC") == canonicalize("CCO") == "CCO"
    assert canonicalize("not a smiles") is None
    assert canonicalize("") is None
    assert canonicalize(None) is None


def test_regression_basic(reg_csv):
    d = load_training_data(reg_csv, "smiles", "y", "regression")
    assert len(d.smiles) == 10 and d.y.dtype == float
    assert d.n_dropped_invalid == 0 and d.n_duplicates_merged == 0


def test_invalid_and_nan_dropped(tmp_path):
    p = write(tmp_path, [("CCO", 1.0), ("garbage", 2.0), ("CCC", None), ("CCN", 3.0)])
    d = load_training_data(p, "smiles", "y", "regression")
    assert d.smiles == ["CCO", "CCN"] and d.n_dropped_invalid == 1


def test_regression_duplicates_mean(tmp_path):
    p = write(tmp_path, [("CCO", 1.0), ("OCC", 3.0), ("CCN", 5.0)])
    d = load_training_data(p, "smiles", "y", "regression")
    assert dict(zip(d.smiles, d.y)) == {"CCO": 2.0, "CCN": 5.0}
    assert d.n_duplicates_merged == 1


def test_classification_duplicates_majority(tmp_path):
    p = write(tmp_path, [("CCO", "a"), ("OCC", "a"), ("CCO", "b"), ("CCN", "b")])
    d = load_training_data(p, "smiles", "y", "classification")
    assert dict(zip(d.smiles, d.y)) == {"CCO": "a", "CCN": "b"}


def test_classification_int_labels_become_str(tmp_path):
    p = write(tmp_path, [("CCO", 0), ("CCN", 1), ("CCC", 1)])
    d = load_training_data(p, "smiles", "y", "classification")
    assert set(d.y) == {"0", "1"}


def test_single_class_rejected(tmp_path):
    p = write(tmp_path, [("CCO", "a"), ("CCN", "a")])
    with pytest.raises(DataError, match="two distinct classes"):
        load_training_data(p, "smiles", "y", "classification")


def test_missing_columns(reg_csv):
    with pytest.raises(DataError, match="not found"):
        load_training_data(reg_csv, "smiles", "nope", "regression")


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_training_data(tmp_path / "x.csv", "smiles", "y", "regression")


def test_empty_file(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("")
    with pytest.raises(DataError, match="empty"):
        load_training_data(p, "smiles", "y", "regression")


def test_non_numeric_regression_target(tmp_path):
    p = write(tmp_path, [("CCO", "high"), ("CCN", "low")])
    with pytest.raises(DataError, match="not numeric"):
        load_training_data(p, "smiles", "y", "regression")


def test_too_few_rows(tmp_path):
    p = write(tmp_path, [("CCO", 1.0), ("bad", 2.0)])
    with pytest.raises(DataError, match="at least"):
        load_training_data(p, "smiles", "y", "regression")
