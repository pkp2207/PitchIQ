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

# Training variants
python -m src.models.train --tune                    # Optuna tuning (30 trials/model) then train
python -m src.models.train --tune --tune-trials 50   # More trials
python -m src.models.train --params models/tuned_params.json  # Use saved tuned params
python -m src.models.train --no-calibrate            # Skip probability calibration
python -m src.models.train --ensemble                # Build soft-voting ensemble from top models
python -m src.models.train --tune --ensemble         # Tune, then build ensemble from best

# Standalone tuning
python -m src.models.tune                            # Tune all models
python -m src.models.tune --model HistGradientBoosting --n-trials 50
python -m src.models.tune --save                     # Save results to models/tuned_params.json

# Run tests
pytest tests/ -v

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

## Architecture

### Data Flow

Raw CSV (`data/raw/results.csv`) -> `cleaner.py` -> `pipeline.py` (orchestrates elo/form/h2h/streak) -> `features.csv` -> `tune.py` (optional) -> `train.py` -> `best_model.pkl` + `thresholds.json` -> `predict.py` / Streamlit app

### Key Modules

- **`src/data/loader.py`** — Loads raw CSVs. `get_project_root()` resolves the repo root (used everywhere). Looks for datasets in both `data/raw/` and `dataset/` directories.
- **`src/data/cleaner.py`** — Filters matches by cutoff year (default 1990), normalizes the `neutral` column, encodes target: Win=1, Draw=0, Loss=-1.
- **`src/features/pipeline.py`** — Orchestrator that runs all feature modules in sequence: Elo -> Form -> H2H -> Streak. Adds `neutral` (int) and `is_friendly` binary features. Outputs 29 feature columns + target.
- **`src/features/elo.py`** — `EloSystem` class iterates chronologically, tracking per-team ratings (base 1500, K=32). Produces `home_elo_before`, `away_elo_before`, `elo_diff`.
- **`src/features/form.py`** — Rolling form over last 5 and 10 matches per team. Stacks home/away into team-level rows, computes `shift(1).rolling()` to avoid data leakage, then merges back. Includes rolling goal difference average (`goal_diff_avg_5`, `goal_diff_avg_10`) and `home_advantage` (rolling win rate over last 10 home matches).
- **`src/features/head_to_head.py`** — Iterates chronologically, accumulating matchup history. Uses `tuple(sorted([home, away]))` as matchup key. Tracks H2H win rate, goal diff, number of meetings.
- **`src/features/streak.py`** — Computes win/loss streak and days-since-last-match for each team. Iterates chronologically over a stacked team-level view. Produces `home_streak`/`away_streak` (positive = consecutive wins, negative = consecutive losses) and `home_days_since_last`/`away_days_since_last` (calendar days since last match, default 30 for first match).
- **`src/models/tune.py`** — Optuna hyperparameter tuning. Defines search spaces for LogisticRegression, RandomForest, GradientBoosting, and HistGradientBoosting. Uses the same temporal validation split as `train.py`. Can run standalone or be invoked by `train.py --tune`. Saves results to `models/tuned_params.json`.
- **`src/models/train.py`** — Trains up to 6 models: LogisticRegression (with StandardScaler), RandomForest, GradientBoosting, HistGradientBoosting, XGBoost, and LightGBM. XGBoost/LightGBM are optional (graceful `except Exception` fallback if libomp not installed). Selects best by macro F1 on temporal validation split. Supports `--tune` for Optuna tuning, `--params` for loading saved params, `--no-calibrate` to skip probability calibration, and `--ensemble` to build a soft-voting ensemble from top models. Uses `class_weight='balanced'` for LR/RF and `compute_sample_weight('balanced')` for GradientBoosting/HistGradientBoosting to address class imbalance. After selection, wraps best model in `CalibratedClassifierCV` (sigmoid for <500 samples, isotonic for larger; skipped for ensembles). Runs per-class threshold optimization on the validation set and saves `thresholds.json` alongside the model. XGBoost labels are mapped from {-1,0,1} to {0,1,2} via `_XGBWrapper`.
- **`src/models/threshold.py`** — Per-class decision threshold tuning. Instead of argmax on raw probabilities, scales each class's probability by a learned factor and then takes argmax. Grid search over scaling factors (wider range for draw class). `optimize_thresholds()` returns optimal thresholds + macro F1; `apply_thresholds()` applies them at inference time.
- **`src/models/ensemble.py`** — `SoftVotingEnsemble` class that averages `predict_proba()` outputs from multiple already-fitted models. Unlike sklearn's `VotingClassifier`, takes pre-fitted models and avoids sample_weight/Pipeline compatibility issues. Provides averaged `feature_importances_` for tree-based sub-models (skips linear models). Used by `train.py --ensemble` when multiple models score within 0.05 macro F1 of the best.
- **`src/models/evaluate.py`** — Evaluates on holdout set. Reports both raw (argmax) and threshold-adjusted metrics: Accuracy, Macro F1, confusion matrix, per-class precision/recall/F1. Also reports Log Loss, Brier Score, calibration status, and reliability diagram data (per-class calibration curves).
- **`src/models/predict.py`** — `MatchPredictor` class loads serialized model + feature schema + `thresholds.json`. Unwraps `CalibratedClassifierCV`, `SoftVotingEnsemble`, `Pipeline`, and `_XGBWrapper` to access the raw estimator for SHAP. Uses `TreeExplainer` for tree models, `LinearExplainer` for LR. Falls back to basic `feature_importances_`/`coef_` if SHAP unavailable. Applies per-class threshold scaling via `apply_thresholds()` before returning predictions. Returns outcome, threshold-adjusted probabilities, and top 5 signed SHAP values. Also provides `get_h2h_history()` for H2H table display.
- **`src/utils/helpers.py`** — `normalize_team_name()` maps aliases to canonical names. `tournament_to_weight()` returns importance weight [0, 1].
- **`app/streamlit_app.py`** — Main UI. Uses `@st.cache_resource` for model, `@st.cache_data` for team list. Calls component modules for rendering.
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
- **Probability calibration** — `train.py` wraps the best model in `CalibratedClassifierCV` by default. `predict.py` unwraps it (via `calibrated_classifiers_[0].estimator`) to access the raw model for SHAP, while still using the calibrated wrapper for `predict_proba()`.
- **Model unwrapping chain** — `predict.py._get_underlying_model()` unwraps in order: `CalibratedClassifierCV` -> `SoftVotingEnsemble` -> `Pipeline` -> `_XGBWrapper` to reach the raw sklearn/xgboost estimator.
- **Optional dependencies** — XGBoost and LightGBM imports use `except Exception` (not just `ImportError`) because their native library load failures raise library-specific errors, not ImportError.
- **Per-class threshold tuning** — `threshold.py` scales each class's raw probability by a learned factor before taking argmax. The draw class (0) gets a wider search range (0.5-3.0) since it is typically under-predicted. Thresholds are optimized on the validation set during training and saved to `models/thresholds.json`. `predict.py` and `evaluate.py` both load and apply thresholds at inference time.
- **Class weighting for imbalance** — Draw is the rarest class. `train.py` uses `class_weight='balanced'` for LogisticRegression and RandomForest, and `compute_sample_weight('balanced')` for GradientBoosting and HistGradientBoosting (which don't support `class_weight` in their constructor). The `SAMPLE_WEIGHT_MODELS` set tracks which models need sample_weight passed to `.fit()`.
- **Soft-voting ensemble** — `train.py --ensemble` builds a `SoftVotingEnsemble` from all models scoring within 0.05 macro F1 of the best individual model. The ensemble averages `predict_proba()` outputs. Calibration is skipped for ensembles since probability averaging across diverse model types is self-calibrating. The ensemble is only used if it beats or ties the best individual model on validation.

### Data Requirements

The raw dataset `results.csv` must exist at `data/raw/results.csv` or `dataset/results.csv`. It can come from the Kaggle dataset "martj42/international-football-results-from-1872-to-2017" or be generated with `scripts/generate_sample_data.py` for development. All processed data and model artifacts are gitignored and regenerated by the pipeline.
