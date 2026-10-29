import numpy as np
import pytest

from moltab.featurizer import N_FEATURES, CheMeleonFeaturizer


def test_fake_contract(featurizer):
    X = featurizer.transform(["CCO", "bad", "CCN"])
    assert X.shape == (3, featurizer.n_features) and X.dtype == np.float32
    assert np.isnan(X[1]).all() and not np.isnan(X[[0, 2]]).any()


def test_empty_input(featurizer):
    assert featurizer.transform([]).shape == (0, featurizer.n_features)


class _Graph:
    pass


def test_chemeleon_batching_and_order(monkeypatch):
    """Real class, patched encoder: checks batching, ordering and NaN rows."""
    import chemprop.data

    f = CheMeleonFeaturizer.__new__(CheMeleonFeaturizer)
    f.batch_size = 2

    class T:
        no_grad = staticmethod(lambda: __import__("contextlib").nullcontext())

    f._torch = T
    f.mol_featurizer = lambda mol: mol.GetNumAtoms()  # "graph" = atom count

    class Model:
        device = "cpu"
        batches = []

        def fingerprint(self, bmg):
            Model.batches.append(list(bmg.graphs))
            arr = np.array(bmg.graphs, dtype=np.float32)[:, None] * np.ones(
                (1, N_FEATURES), np.float32
            )

            class W:
                def numpy(self, force=True):
                    return arr

            return W()

    class FakeBMG:
        def __init__(self, graphs):
            self.graphs = graphs

        def to(self, device):
            pass

    f.model = Model()
    monkeypatch.setattr(chemprop.data, "BatchMolGraph", FakeBMG)
    X = f.transform(["C", "bad", "CC", "CCC", "CCCC"])
    assert [len(b) for b in Model.batches] == [2, 2]
    assert np.isnan(X[1]).all()
    assert X[[0, 2, 3, 4], 0].tolist() == [1, 2, 3, 4]


@pytest.mark.slow
def test_real_chemeleon():
    f = CheMeleonFeaturizer(device="cpu")
    X = f.transform(["CCO", "c1ccccc1", "bad"])
    assert X.shape == (3, N_FEATURES) and np.isnan(X[2]).all()
    np.testing.assert_allclose(X[:2], f.transform(["CCO", "c1ccccc1"]), atol=1e-5)
