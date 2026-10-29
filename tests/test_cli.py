import pytest

from moltab import cli


def parse(argv):
    return cli.settings_from_args(cli.build_parser().parse_args(argv))


def test_args_to_settings():
    s = parse(
        [
            "serve",
            "--data",
            "t.csv",
            "--task",
            "regression",
            "--smiles-col",
            "S",
            "--target-col",
            "T",
            "--n-estimators",
            "4",
            "--port",
            "9001",
            "--no-kv-cache",
        ]
    )
    assert (s.smiles_col, s.target_col, s.n_estimators, s.port, s.kv_cache) == (
        "S",
        "T",
        4,
        9001,
        False,
    )


def test_cli_overrides_env_and_env_fills_gaps(monkeypatch):
    monkeypatch.setenv("MOLTAB_PORT", "7000")
    monkeypatch.setenv("MOLTAB_TARGET_COL", "env_t")
    s = parse(["serve", "--data", "t.csv", "--task", "classification", "--port", "8001"])
    assert s.port == 8001 and s.target_col == "env_t"


def test_invalid_task_rejected_by_parser():
    with pytest.raises(SystemExit):
        parse(["serve", "--data", "t.csv", "--task", "bogus"])


def test_main_runs_uvicorn(monkeypatch, reg_csv):
    import uvicorn

    called = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: called.update(host=host, port=port, app=app)
    )
    cli.main(["serve", "--data", str(reg_csv), "--task", "regression", "--port", "8123"])
    assert called["port"] == 8123 and called["app"] is not None


def test_main_missing_config_exits(monkeypatch):
    monkeypatch.delenv("MOLTAB_DATA_PATH", raising=False)
    monkeypatch.delenv("MOLTAB_TASK", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["serve"])
