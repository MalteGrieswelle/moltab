"""TabICLv2 wrapper: in-context regression / classification on fingerprints."""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

from moltab.config import Task


def default_estimator_factory(task: Task, **kwargs: Any):
    from tabicl import TabICLClassifier, TabICLRegressor

    cls = TabICLRegressor if task == "regression" else TabICLClassifier
    return cls(**kwargs)


class ScoringModel:
    def __init__(
        self,
        task: Task,
        n_estimators: int = 8,
        device: str | None = None,
        kv_cache: bool = True,
        estimator_factory: Callable[..., Any] = default_estimator_factory,
    ):
        self.task = task
        self._est = estimator_factory(
            task, n_estimators=n_estimators, device=device, kv_cache=kv_cache
        )
        self._labels: LabelEncoder | None = None
        self._fitted = False

    @property
    def classes(self) -> list[str] | None:
        return None if self._labels is None else [str(c) for c in self._labels.classes_]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScoringModel":
        if self.task == "classification":
            self._labels = LabelEncoder().fit(y)
            y = self._labels.transform(y)
        self._est.fit(X, y)
        self._fitted = True
        return self

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Regression: values. Classification: predicted class labels."""
        self._check()
        pred = self._est.predict(X)
        return (
            self._labels.inverse_transform(pred)
            if self._labels is not None
            else np.asarray(pred, float)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, n_classes), columns ordered as `classes`."""
        self._check()
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification")
        return np.asarray(self._est.predict_proba(X), dtype=float)
