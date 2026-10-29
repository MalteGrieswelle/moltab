import pytest
from fastapi.testclient import TestClient
from test_api import app_for

from moltab.client import MoltabClient


@pytest.fixture
def client(featurizer, estimator_factory, reg_csv):
    with TestClient(app_for(featurizer, estimator_factory, reg_csv, "regression")) as tc:
        yield MoltabClient("http://testserver", session=_Adapter(tc))


class _Adapter:
    """requests-like session on top of the TestClient (which wants relative paths)."""

    def __init__(self, tc):
        self.tc = tc

    def post(self, url, json, timeout):
        return self.tc.post(url.replace("http://testserver", ""), json=json)

    def get(self, url, timeout):
        return self.tc.get(url.replace("http://testserver", ""))


def test_score_str_returns_float(client):
    assert isinstance(client.score("CCO"), float)


def test_score_list_with_invalid(client):
    s = client.score(["CCO", "bad"])
    assert isinstance(s[0], float) and s[1] is None


def test_score_full_and_info(client):
    assert client.score_full("CCO")[0]["valid"] is True
    assert client.info()["task"] == "regression"


def test_http_error_raised(client):
    with pytest.raises(Exception):
        client.score([])
