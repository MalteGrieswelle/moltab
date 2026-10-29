"""FastAPI application."""

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from moltab.config import Settings
from moltab.schemas import InfoResponse, ScoreRequest, ScoreResponse
from moltab.service import Scorer, build_scorer


def create_app(
    settings: Settings | None = None,
    scorer_factory: Callable[[Settings], Scorer] = build_scorer,
    max_request_size: int | None = None,
) -> FastAPI:
    """The scorer is built once at startup (lifespan) so bad data fails fast."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = scorer_factory(settings or Settings())
        yield

    limit = max_request_size or (settings.max_request_size if settings else 1000)
    app = FastAPI(title="Moltab", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse)
    def info():
        s: Scorer = app.state.scorer
        return InfoResponse(
            task=s.task,
            n_train=s.n_train,
            n_features=s.featurizer.n_features,
            device=s.device,
            classes=s.model.classes,
            positive_class=s.positive_class,
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        if len(req.smiles) > limit:
            raise HTTPException(422, f"At most {limit} SMILES per request, got {len(req.smiles)}")
        return ScoreResponse(results=app.state.scorer.score(req.smiles))

    return app
