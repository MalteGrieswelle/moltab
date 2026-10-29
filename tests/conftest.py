import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression, Ridge

from moltab.config import Settings

REG_SMILES = [
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "c1ccccc1C",
    "CC(=O)O",
    "CCN",
    "CCCN",
    "CCCCN",
    "c1ccccc1O",
]
REG_Y = [1.0, 2.0, 3.0, 5.0, 5.5, 0.5, 1.5, 2.5, 3.5, 4.5]


class FakeFeaturizer:
    """Morgan bits instead of CheMeleon; NaN rows for invalid SMILES."""

    n_features = 128
    device = "cpu"

    def __init__(self):
        self.calls = 0
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.n_features)

    def transform(self, smiles):
        self.calls += 1
        out = np.full((len(smiles), self.n_features), np.nan, dtype=np.float32)
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if mol is not None:
                out[i] = self.gen.GetFingerprintAsNumPy(mol)
        return out


class FakeEstimator:
    """sklearn stand-in with the TabICL constructor signature."""

    def __init__(self, task, **kwargs):
        self.task, self.kwargs = task, kwargs
        self.est = (
            Ridge(alpha=0.1) if task == "regression" else LogisticRegression(max_iter=500, C=10)
        )
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict(X)

    def predict_proba(self, X):
        return self.est.predict_proba(X)


@pytest.fixture
def featurizer():
    return FakeFeaturizer()


@pytest.fixture
def reg_csv(tmp_path):
    p = tmp_path / "reg.csv"
    pd.DataFrame({"smiles": REG_SMILES, "y": REG_Y}).to_csv(p, index=False)
    return p


@pytest.fixture
def clf_csv(tmp_path):
    p = tmp_path / "clf.csv"
    labels = ["alcohol" if "O" in s and "c" not in s else "other" for s in REG_SMILES]
    pd.DataFrame({"smiles": REG_SMILES, "y": labels}).to_csv(p, index=False)
    return p


def make_settings(path, task, **kw):
    return Settings(data_path=path, task=task, smiles_col="smiles", target_col="y", **kw)


@pytest.fixture
def estimator_factory():
    return FakeEstimator
