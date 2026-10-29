import pytest
from pydantic import ValidationError

from moltab.schemas import ScoreRequest


def test_str_coerced():
    assert ScoreRequest(smiles="CCO").smiles == ["CCO"]


def test_list_kept():
    assert ScoreRequest(smiles=["CCO", "CC"]).smiles == ["CCO", "CC"]


def test_empty_rejected():
    with pytest.raises(ValidationError):
        ScoreRequest(smiles=[])


def test_wrong_type():
    with pytest.raises(ValidationError):
        ScoreRequest(smiles=[1, 2])
