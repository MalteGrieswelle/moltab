"""REST request/response models."""

from pydantic import BaseModel, field_validator


class ScoreRequest(BaseModel):
    smiles: list[str]

    @field_validator("smiles", mode="before")
    @classmethod
    def _single_to_list(cls, v):
        return [v] if isinstance(v, str) else v

    @field_validator("smiles")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("smiles must contain at least one entry")
        return v


class ScoreItem(BaseModel):
    smiles: str
    valid: bool
    score: float | None = None
    label: str | None = None  # classification only
    probabilities: dict[str, float] | None = None  # classification only


class ScoreResponse(BaseModel):
    results: list[ScoreItem]


class InfoResponse(BaseModel):
    task: str
    n_train: int
    n_features: int
    device: str | None
    classes: list[str] | None
    positive_class: str | None
