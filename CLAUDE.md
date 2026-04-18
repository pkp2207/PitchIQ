# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PitchIQ is an ML pipeline that predicts international football match outcomes (Win/Draw/Loss from the home team's perspective) using historical FIFA match data and engineered features. It exposes predictions through a Streamlit web app.

The project is built in phases: Phase 1 (core pipeline + Streamlit) is implemented, Phase 2 (Transfermarkt squad strength) and Phase 3 (sentiment analysis) are planned.

## Commands

```bash
# Install the project (editable mode, sets up imports without sys.path hacks)
pip install -e .

# Generate synthetic sample data (no Kaggle credentials needed)
python scripts/generate_sample_data.py

# Download real datasets (requires Kaggle API credentials)
python scripts/download_data.py

# Run the full pipeline end-to-end
python -m src.data.cleaner              # Clean raw data -> data/processed/matches_cleaned.csv
python -m src.features.pipeline         # Feature engineering -> data/processed/features.csv
python -m src.models.train              # Train + select model -> models/best_model.pkl
python -m src.models.evaluate           # Evaluate on holdout set -> models/evaluation_report.json

# Run tests
pytest tests/ -v

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

## Architecture

### Data Flow

Raw CSV (`data/raw/results.csv`) -> `cleaner.py` -> `pipeline.py` (orchestrates elo/form/h2h) -> `features.csv` -> `train.py` -> `best_model.pkl` -> `predict.py` / Streamlit app

### Key Modules

- **`src/data/loader.py`** — Loads raw CSVs. `get_project_root()` resolves the repo root (used everywhere). Looks for datasets in both `data/raw/` and `dataset/` directories.
- **`src/data/cleaner.py`** — Filters matches by cutoff year (default 1990), normalizes the `neutral` column, encodes target: Win=1, Draw=0, Loss=-1.
- **`src/features/pipeline.py`** — Orchestrator that runs all feature modules in sequence: Elo -> Form -> H2H. Adds `neutral` (int) and `is_friendly` binary features. Outputs 20 feature columns + target.
- **`src/features/elo.py`** — `EloSystem` class iterates chronologically, tracking per-team ratings (base 1500, K=32). Produces `home_elo_before`, `away_elo_before`, `elo_diff`.
- **`src/features/form.py`** — Rolling form over last 5 and 10 matches per team. Stacks home/away into team-level rows, computes `shift(1).rolling()` to avoid data leakage, then merges back.
- **`src/features/head_to_head.py`** — Iterates chronologically, accumulating matchup history. Uses `tuple(sorted([home, away]))` as matchup key. Tracks H2H win rate, goal diff, number of meetings.
- **`src/models/train.py`** — Trains LogisticRegression (with StandardScaler), RandomForest, XGBoost, and LightGBM. XGBoost/LightGBM are optional (graceful fallback if libomp not installed). Selects best by macro F1 on a temporal validation split. XGBoost labels are mapped from {-1,0,1} to {0,1,2} via `_XGBWrapper`.
- **`src/models/evaluate.py`** — Evaluates on holdout set. Reports Accuracy, Macro F1, Log Loss, Brier Score, confusion matrix, and classification report.
- **`src/models/predict.py`** — `MatchPredictor` class loads serialized model + feature schema. Uses SHAP (`TreeExplainer` or `LinearExplainer`) for per-prediction feature attribution. Falls back to basic `feature_importances_`/`coef_` if SHAP unavailable. Returns outcome, probabilities, and top 5 signed SHAP values. Also provides `get_h2h_history()` for H2H table display.
- **`src/utils/helpers.py`** — `normalize_team_name()` maps aliases to canonical names. `tournament_to_weight()` returns importance weight [0, 1].
- **`app/streamlit_app.py`** — Main UI. Uses `@st.cache_resource` for model, `@st.cache_data` for team list.
- **`app/components/prediction_card.py`** — Outcome badge (colored) + probability stacked bar.
- **`app/components/feature_attribution.py`** — Horizontal SHAP bar chart (teal=positive, orange=negative).
- **`app/components/h2h_table.py`** — Last 5 H2H encounters table + win/draw/loss tally metrics.

### Important Patterns

- **Temporal ordering is critical** — All feature computation (Elo, form, H2H) iterates chronologically and only uses past data. The train/test split is also temporal (not random). Breaking chronological order causes data leakage.
- **`pyproject.toml` with editable install** — Use `pip install -e .` to make `src` importable. No more `sys.path` hacking in pipeline modules.
- **Target encoding** — Outcome is always from the home team's perspective: 1 (win), 0 (draw), -1 (loss). Away team outcomes are reversed in `form.py`.
- **XGBoost label mapping** — XGBoost requires non-negative labels. `_XGBWrapper` in `train.py` maps {-1,0,1} to {0,1,2} for training and maps back for prediction.
- **Feature columns are tracked** — `models/feature_columns.json` is committed to Git and defines the exact feature schema the model expects at inference time.
- **NaN handling** — The pipeline fills NaN with 0 (`fillna(0)`) for teams with insufficient history. This is intentional for early matches in the dataset.
- **SHAP explanations** — `predict.py` uses TreeExplainer for tree models and LinearExplainer for logistic regression. Returns signed SHAP values (positive = pushes toward predicted class).

### Data Requirements

The raw dataset `results.csv` must exist at `data/raw/results.csv` or `dataset/results.csv`. It can come from the Kaggle dataset "martj42/international-football-results-from-1872-to-2017" or be generated with `scripts/generate_sample_data.py` for development. All processed data and model artifacts are gitignored and regenerated by the pipeline.
