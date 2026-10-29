"""`moltab serve` command."""

import argparse
import sys

from pydantic import ValidationError

from moltab.config import Settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="moltab")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="start the scoring server")
    s.add_argument("--data", dest="data_path", help="training CSV")
    s.add_argument("--task", choices=["regression", "classification"])
    s.add_argument("--smiles-col")
    s.add_argument("--target-col")
    s.add_argument("--positive-class")
    s.add_argument("--n-estimators", type=int)
    s.add_argument("--device")
    s.add_argument("--no-kv-cache", dest="kv_cache", action="store_const", const=False)
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    """CLI flags override MOLTAB_* env vars."""
    overrides = {k: v for k, v in vars(args).items() if k != "command" and v is not None}
    return Settings(**overrides)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        settings = settings_from_args(args)
    except ValidationError as exc:
        sys.exit(f"Invalid configuration:\n{exc}")

    import uvicorn

    from moltab.api import create_app

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
