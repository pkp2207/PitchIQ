"""Optuna-based hyperparameter tuning over the temporal validation split.

Usage:
    python -m src.models.tune                  # tune all available models
    python -m src.models.tune --model HistGradientBoosting
    python -m src.models.tune --n-trials 50
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from src.data.loader import get_project_root

# Models that need sample_weight in .fit() rather than class_weight in constructor
SAMPLE_WEIGHT_MODELS = {'GradientBoosting', 'HistGradientBoosting'}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_search_spaces():
    """Return a dict of model_name -> (create_model_fn, param_space_fn) tuples.

    Each param_space_fn takes an optuna.Trial and returns a model instance.
    """
    spaces = {}

    def _lr(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, class_weight='balanced',
                                       max_iter=2000, random_state=42)),
        ])
    spaces["LogisticRegression"] = _lr

    def _rf(trial):
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight='balanced',
            random_state=42,
        )
    spaces["RandomForest"] = _rf

    def _gb(trial):
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            random_state=42,
        )
    spaces["GradientBoosting"] = _gb

    def _hgb(trial):
        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("max_iter", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
            random_state=42,
        )
    spaces["HistGradientBoosting"] = _hgb

    return spaces


def _load_data():
    root = get_project_root()
    data_path = root / "data" / "processed" / "features.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Features not found at {data_path}. Run the pipeline first.")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    exclude = ["date", "home_team", "away_team", "tournament", "outcome"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y = df["outcome"]

    # 90/10 temporal holdout — same as train.py
    split_idx = int(len(df) * 0.9)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    # 80/20 temporal validation within training set — same as train.py
    val_idx = int(len(X_train) * 0.8)
    X_t, X_v = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
    y_t, y_v = y_train.iloc[:val_idx], y_train.iloc[val_idx:]

    # Precompute balanced sample weights for models that need them
    sw_t = compute_sample_weight('balanced', y_t)

    return X_t, y_t, X_v, y_v, feature_cols, sw_t


def tune_model(model_name: str, n_trials: int = 30) -> dict:
    """Run Optuna tuning for a single model. Returns best params + score."""
    spaces = _get_search_spaces()
    if model_name not in spaces:
        raise ValueError(f"Unknown model {model_name}. Available: {list(spaces.keys())}")

    X_t, y_t, X_v, y_v, _, sw_t = _load_data()
    create_fn = spaces[model_name]

    def objective(trial):
        model = create_fn(trial)
        if model_name in SAMPLE_WEIGHT_MODELS:
            model.fit(X_t, y_t, sample_weight=sw_t)
        else:
            model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return f1_score(y_v, preds, average="macro")

    study = optuna.create_study(direction="maximize", study_name=model_name)
    study.optimize(objective, n_trials=n_trials)

    return {
        "model": model_name,
        "best_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
    }


def tune_all(n_trials: int = 30) -> list[dict]:
    """Tune all available models and return results sorted by best F1."""
    spaces = _get_search_spaces()
    results = []
    for name in spaces:
        print(f"\nTuning {name} ({n_trials} trials)...")
        result = tune_model(name, n_trials)
        print(f"  Best F1: {result['best_f1']:.4f}  Params: {result['best_params']}")
        results.append(result)

    results.sort(key=lambda r: r["best_f1"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna.")
    parser.add_argument("--model", type=str, default=None,
                        help="Tune a specific model (e.g., HistGradientBoosting). Default: tune all.")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials per model (default: 30).")
    parser.add_argument("--save", action="store_true",
                        help="Save best params to models/tuned_params.json.")
    args = parser.parse_args()

    if args.model:
        results = [tune_model(args.model, args.n_trials)]
        print(f"\n{args.model} — Best F1: {results[0]['best_f1']:.4f}")
        print(f"  Params: {results[0]['best_params']}")
    else:
        results = tune_all(args.n_trials)
        print("\n" + "=" * 60)
        print("TUNING RESULTS (sorted by F1):")
        for r in results:
            print(f"  {r['model']:25s}  F1={r['best_f1']:.4f}")
        print("=" * 60)

    if args.save:
        root = get_project_root()
        out_path = root / "models" / "tuned_params.json"
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved tuning results to {out_path}")


if __name__ == "__main__":
    main()
