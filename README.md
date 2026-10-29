# Moltab

Moltab is a scoring function using the TabICLv2 tabular foundation model paired with the CheMeleon foundation model fingerprints to predict molecular objectives.
It spinns up a `Docker` container with a training dataset to set the context for the TabICL model.
Then the user can query the model with a SMILES string via a REST API query and gets back the regression or classification score.

## Spinning Up The Server

If you spinn up the server you have to provide a dataset (CSV) containing SMILES strings and target values.
You also have to set if it is a regression task or a classification task.
After starting the docker server you can query the scoring function with pythons requests.
You send a SMILES string or a list of SMILES and get a prediction back.

## Installation
```bash
# only installs the client dependencies
# you can query the server
pip install moltab

# to host a local docker server use the extra dependencies
pip install "moltab[server]"
```

## Usage

Training CSV needs a SMILES column and a target column (numeric for regression, any label for classification).
Invalid SMILES are dropped, duplicates merged (mean / majority vote).
Features are the raw 2048-dim CheMeleon fingerprint.

```bash
# local
uv run moltab serve --data data/train.csv --task regression --smiles-col smiles --target-col target
# docker (GPU if available via nvidia runtime; edit MOLTAB_* in docker-compose.yml)
docker compose up --build
```

Settings can also be given as `MOLTAB_*` env vars (`DATA_PATH`, `TASK`, `SMILES_COL`, `TARGET_COL`, `POSITIVE_CLASS`, `N_ESTIMATORS`, `DEVICE`, `KV_CACHE`, `PORT`, ...); CLI flags win.

```python
from moltab.client import MoltabClient

c = MoltabClient("http://localhost:8000") # default adress of the local server
c.score("CCO")                 # -> float
c.score(["CCO", "c1ccccc1"])   # -> list, None for invalid SMILES
c.score_full(["CCO"])          # -> dicts incl. label/probabilities for classification
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

Two datasets have been tested (solubility from Biogen ADME Fang et al. and cyp3a4 from Novartis). 
The performance is very much comparable with a MLP trained on CheMeleon fingerprints to predict these log-transformed outcomes.
