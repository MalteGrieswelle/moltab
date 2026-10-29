"""Minimal Python client for the Moltab REST API."""

import requests


class MoltabClient:
    def __init__(self, url: str = "http://localhost:8000", timeout: float = 120.0, session=None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests

    def _post(self, smiles) -> list[dict]:
        r = self.session.post(f"{self.url}/score", json={"smiles": smiles}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["results"]

    def info(self) -> dict:
        r = self.session.get(f"{self.url}/info", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def score_full(self, smiles: str | list[str]) -> list[dict]:
        return self._post(smiles)

    def score(self, smiles: str | list[str]) -> float | None | list[float | None]:
        """Score(s) only; None for invalid SMILES. A str in -> a float out."""
        results = self._post(smiles)
        scores = [r["score"] for r in results]
        return scores[0] if isinstance(smiles, str) else scores
