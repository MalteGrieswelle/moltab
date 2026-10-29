# Moltab

Moltab is a scoring function using the [TabICLv2](https://github.com/soda-inria/tabicl) tabular foundation model paired with the [CheMeleon](https://github.com/JacksonBurns/chemeleon) foundation model embeddings to predict molecular objectives.
It spinns up a local server with a training dataset to set the in context for the TabICL model.
Then the user can query the server with a SMILES string (or list of SMIELS) via its API and gets back the regression or classification score.

## Installation
```bash
# only installs the client dependencies
# you can query the server
pip install moltab

# to host a local docker server use the extra dependencies
pip install "moltab[server]"
```

## Usage

The training/ context CSV needs a SMILES column and a target column (numeric for regression, any label for classification).
Invalid SMILES are dropped, duplicates merged (mean / majority vote).
Features are the raw 2048-dim CheMeleon fingerprint.

```bash
# local (pip/uv)
uv run moltab serve --data data/train.csv --task regression --smiles-col smiles --target-col target

# Optional: docker from a clone of the GitHub repo
docker compose up --build
```

Settings can also be given as `MOLTAB_*` env vars (`DATA_PATH`, `TASK`, `SMILES_COL`, `TARGET_COL`, `POSITIVE_CLASS`, `N_ESTIMATORS`, `DEVICE`, `KV_CACHE`, `PORT`, ...); CLI flags win.

```python
from moltab.client import MoltabClient

c = MoltabClient("http://localhost:8000") # default adress of the local server
c.score("CCO")                 # returns float
c.score(["CCO", "c1ccccc1"])   # returns list, None for invalid SMILES
c.score_full(["CCO"])          # returns dicts incl. label/probabilities for classification
```

Endpoints: `GET /health`, `GET /info`, `POST /score {"smiles": str | [str]}`.
For binary classification `score` is the probability of `positive_class` (default: last class alphabetically).

## Tests

`uv run pytest` (fast, fakes only); `uv run pytest -m slow` (real CheMeleon/TabICL weights).

## Evaluating prediction quality

```bash
uv run python scripts/cross_validate.py data/train.csv --smiles-col smiles --target-col target
```
Runs 5x5 repeated k-fold CV (25 folds, fresh model per fold, fingerprints computed once), prints mean/std/median of R2, MAE, MSE and Kendall tau, and saves a boxplot to `<csv>_cv.png`. Regression only.

Two datasets have been tested ([solubility from Biogen ADME Fang et al.](https://doi.org/10.1021/acs.jcim.3c00160) and [cyp3a4 from Novartis](https://doi.org/10.1021/acs.chemrestox.3c00305
)). 
The performance is very much comparable with an MLP trained on CheMeleon fingerprints to predict these log-transformed outcomes.
