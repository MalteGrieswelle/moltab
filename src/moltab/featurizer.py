"""SMILES -> CheMeleon fingerprint."""

from typing import Protocol

import numpy as np
from rdkit import Chem

N_FEATURES = 2048
CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


class Featurizer(Protocol):
    n_features: int

    def transform(self, smiles: list[str]) -> np.ndarray:
        """(n, n_features) float32 array. Rows of unparsable SMILES are NaN."""
        ...


class CheMeleonFeaturizer:
    """CheMeleon message-passing encoder (adapted from JacksonBurns/chemeleon, MIT)."""

    n_features = N_FEATURES

    def __init__(self, device: str | None = None, batch_size: int = 256, checkpoint_dir=None):
        from pathlib import Path
        from urllib.request import urlretrieve

        import torch
        from chemprop import featurizers, nn
        from chemprop.models import MPNN
        from chemprop.nn import RegressionFFN

        self._torch = torch
        self.batch_size = batch_size
        self.mol_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path.home() / ".chemprop"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            urlretrieve(CHECKPOINT_URL, mp_path)
        ckpt = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        self.model = MPNN(
            message_passing=mp,
            agg=nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        self.model.eval()
        self.model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @property
    def device(self) -> str:
        return str(self.model.device)

    def transform(self, smiles: list[str]) -> np.ndarray:
        from chemprop.data import BatchMolGraph

        out = np.full((len(smiles), self.n_features), np.nan, dtype=np.float32)
        mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]
        idx = [i for i, m in enumerate(mols) if m is not None]
        for start in range(0, len(idx), self.batch_size):
            chunk = idx[start : start + self.batch_size]
            bmg = BatchMolGraph([self.mol_featurizer(mols[i]) for i in chunk])
            bmg.to(device=self.model.device)
            with self._torch.no_grad():
                out[chunk] = self.model.fingerprint(bmg).numpy(force=True)
        return out
