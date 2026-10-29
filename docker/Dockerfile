FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    HOME=/home/moltab PATH=/app/.venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --extra server --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra server
# Pre-download weights so the container runs offline.
RUN mkdir -p $HOME && python -c "\
from moltab.featurizer import CheMeleonFeaturizer; CheMeleonFeaturizer(device='cpu'); \
from tabicl import TabICLRegressor, TabICLClassifier; \
TabICLRegressor(device='cpu'); TabICLClassifier(device='cpu')" \
 && python - <<'PY'
import numpy as np
from tabicl import TabICLRegressor, TabICLClassifier
X = np.random.rand(40, 4); TabICLRegressor(device="cpu", n_estimators=1).fit(X, X[:, 0]).predict(X)
TabICLClassifier(device="cpu", n_estimators=1).fit(X, (X[:, 0] > .5).astype(int)).predict(X)
PY
RUN chmod -R a+rwX /home/moltab
EXPOSE 8000
ENTRYPOINT ["moltab", "serve"]
