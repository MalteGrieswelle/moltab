"""Scorer: featurize -> predict, handling invalid SMILES and request limits."""

import threading

import numpy as np

from moltab.config import Settings
from moltab.data import TrainingData
from moltab.featurizer import Featurizer
from moltab.model import ScoringModel
from moltab.schemas import ScoreItem


class Scorer:
    def __init__(
        self,
        featurizer: Featurizer,
        model: ScoringModel,
        train: TrainingData,
        positive_class: str | None = None,
        device: str | None = None,
    ):
        self.featurizer = featurizer
        self.model = model
        self.n_train = len(train.smiles)
        self.device = device
        self._lock = threading.Lock()  # one GPU model, serialize inference
        X = featurizer.transform(train.smiles)
        keep = ~np.isnan(X).any(axis=1)
        if not keep.all():  # canonical SMILES should always featurize; be safe anyway
            X, y = X[keep], train.y[keep]
        else:
            y = train.y
        self.model.fit(X, y)
        self.positive_class = self._resolve_positive(model.classes, positive_class)

    @staticmethod
    def _resolve_positive(classes, positive_class) -> str | None:
        if classes is None:
            return None
        if positive_class is not None:
            if positive_class not in classes:
                raise ValueError(f"positive_class {positive_class!r} not in classes {classes}")
            return positive_class
        return classes[-1] if len(classes) == 2 else None

    @property
    def task(self) -> str:
        return self.model.task

    def score(self, smiles: list[str]) -> list[ScoreItem]:
        items = [ScoreItem(smiles=s, valid=False) for s in smiles]
        X = self.featurizer.transform(smiles)
        ok = np.flatnonzero(~np.isnan(X).any(axis=1))
        if len(ok) == 0:
            return items
        Xv = X[ok]
        with self._lock:
            if self.task == "regression":
                preds = self.model.predict(Xv)
                for i, p in zip(ok, preds):
                    items[i] = ScoreItem(smiles=smiles[i], valid=True, score=float(p))
            else:
                proba = self.model.predict_proba(Xv)
                classes = self.model.classes
                pos = classes.index(self.positive_class) if self.positive_class else None
                for i, row in zip(ok, proba):
                    top = int(np.argmax(row))
                    items[i] = ScoreItem(
                        smiles=smiles[i],
                        valid=True,
                        score=float(row[pos if pos is not None else top]),
                        label=classes[top],
                        probabilities={c: float(p) for c, p in zip(classes, row)},
                    )
        return items


def build_scorer(
    settings: Settings, featurizer: Featurizer | None = None, model: ScoringModel | None = None
) -> Scorer:
    from moltab.data import load_training_data

    train = load_training_data(
        settings.data_path, settings.smiles_col, settings.target_col, settings.task
    )
    if featurizer is None:
        from moltab.featurizer import CheMeleonFeaturizer

        featurizer = CheMeleonFeaturizer(settings.device, settings.featurize_batch_size)
    if model is None:
        model = ScoringModel(
            settings.task, settings.n_estimators, settings.device, settings.kv_cache
        )
    device = getattr(featurizer, "device", settings.device)
    return Scorer(featurizer, model, train, settings.positive_class, device)
