"""Server settings, read from MOLTAB_* environment variables or passed explicitly."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Task = Literal["regression", "classification"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOLTAB_", extra="forbid")

    data_path: Path
    task: Task
    smiles_col: str = "smiles"
    target_col: str = "target"
    positive_class: str | None = None  # binary classification: class whose probability is `score`
    n_estimators: int = Field(8, ge=1)
    device: str | None = None  # None -> auto (CUDA if available, else CPU)
    kv_cache: bool = True
    featurize_batch_size: int = Field(256, ge=1)
    max_request_size: int = Field(1000, ge=1)
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
