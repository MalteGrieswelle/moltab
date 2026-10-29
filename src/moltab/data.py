"""Loading and cleaning the training table."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from moltab.config import Task

RDLogger.DisableLog("rdApp.*")

MIN_TRAIN_ROWS = 2


class DataError(ValueError):
    """The training data is unusable; the message is meant for the user."""


@dataclass(frozen=True)
class TrainingData:
    smiles: list[str]  # canonical, unique
    y: np.ndarray  # float for regression, str for classification
    n_dropped_invalid: int
    n_duplicates_merged: int


def canonicalize(smiles: str) -> str | None:
    """Canonical SMILES, or None if RDKit cannot parse it."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return None if mol is None else Chem.MolToSmiles(mol)


def _majority(values: pd.Series) -> str:
    return values.value_counts().sort_index().idxmax()  # ties -> alphabetically first


def load_training_data(path: Path, smiles_col: str, target_col: str, task: Task) -> TrainingData:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Training file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"Training file is empty: {path}") from exc
    if missing := [c for c in (smiles_col, target_col) if c not in df.columns]:
        raise DataError(f"Columns {missing} not found in {path}; available: {list(df.columns)}")

    df = df[[smiles_col, target_col]].dropna()
    if task == "regression":
        try:
            target = pd.to_numeric(df[target_col])
        except (ValueError, TypeError) as exc:
            raise DataError(f"Target column {target_col!r} is not numeric") from exc
    else:
        target = df[target_col].astype(str)

    canon = df[smiles_col].map(canonicalize)
    valid = canon.notna()
    n_invalid = int((~valid).sum())
    clean = pd.DataFrame({"smiles": canon[valid], "y": target[valid]})

    grouped = clean.groupby("smiles", sort=False)["y"]
    merged = grouped.mean() if task == "regression" else grouped.agg(_majority)
    n_dups = len(clean) - len(merged)

    if len(merged) < MIN_TRAIN_ROWS:
        raise DataError(
            f"Need at least {MIN_TRAIN_ROWS} valid training molecules, got {len(merged)}"
        )
    if task == "classification" and merged.nunique() < 2:
        raise DataError("Classification needs at least two distinct classes in the target column")

    y = merged.to_numpy(dtype=float if task == "regression" else str)
    return TrainingData(list(merged.index), y, n_invalid, n_dups)
