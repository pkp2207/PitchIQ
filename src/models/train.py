import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from src.data.loader import get_project_root
from src.models.ensemble import SoftVotingEnsemble
from src.models.threshold import optimize_thresholds

# Models that don't support class_weight in the constructor and need
# sample_weight passed to .fit() instead.
SAMPLE_WEIGHT_MODELS = {'GradientBoosting', 'HistGradientBoosting'}


def _compute_sample_weights(y):
    """Compute balanced sample weights so minority classes (draws) get higher weight."""
    return compute_sample_weight('balanced', y)


# XGBoost requires non-negative integer labels.
# Our labels are {-1, 0, 1} so we map them to {0, 1, 2} for training
# and map back after prediction.
LABEL_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_LABEL = {v: k for k, v in LABEL_TO_ENCODED.items()}


def _encode_labels(y):
    return y.map(LABEL_TO_ENCODED)


def _decode_labels(y_encoded):
    return np.array([ENCODED_TO_LABEL[v] for v in y_encoded])


class _XGBWrapper:
    """Wraps XGBClassifier to handle label mapping transparently."""

    def __init__(self, **kwargs):
        self.xgb = XGBClassifier(**kwargs)
        self.classes_ = np.array([-1, 0, 1])

    def fit(self, X, y, sample_weight=None):
        self.xgb.fit(X, _encode_labels(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return _decode_labels(self.xgb.predict(X))

    def predict_proba(self, X):
        # XGB returns probs in encoded order [0,1,2] which maps to [-1,0,1]
        return self.xgb.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.xgb.feature_importances_

    def get_booster(self):
        return self.xgb.get_booster()


def _build_model_from_params(name: str, params: dict):
    """Instantiate a model from its name and tuned parameter dict."""
    if name == 'LogisticRegression':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**params, class_weight='balanced',
                                       max_iter=2000, random_state=42)),
        ])
    elif name == 'RandomForest':
        return RandomForestClassifier(**params, class_weight='balanced', random_state=42)
    elif name == 'GradientBoosting':
        return GradientBoostingClassifier(**params, random_state=42)
    elif name == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(**params, random_state=42)
    elif name == 'XGBoost' and HAS_XGB:
        return _XGBWrapper(**params, use_label_encoder=False, eval_metric='mlogloss',
                           random_state=42, verbosity=0)
    elif name == 'LightGBM' and HAS_LGBM:
        return LGBMClassifier(**params, random_state=42, verbose=-1)
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")


def _default_models():
    """Return dict of models with default hyperparameters."""
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced',
                                       max_iter=2000, random_state=42)),
        ]),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced',
            random_state=42,
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, learning_rate=0.1, random_state=42,
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = _XGBWrapper(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0,
        )

    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1,
        )

    return models


def train_and_select_model(calibrate: bool = True, tune: bool = False,
                           tune_trials: int = 30, params_path: str = None,
                           ensemble: bool = False):
    root = get_project_root()
    data_path = root / "data" / "processed" / "features.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Features file not found at {data_path}. Run src/features/pipeline.py first.")

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Exclude non-numeric and target columns from features
    exclude_cols = ['date', 'home_team', 'away_team', 'tournament', 'outcome']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df['outcome']

    # Train-test split based on time (last 10% of data as holdout test)
    split_idx = int(len(df) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Determine model set: tuned params, loaded params, or defaults
    if tune:
        print(f"\nRunning Optuna hyperparameter tuning ({tune_trials} trials per model)...")
        from src.models.tune import tune_all
        tuning_results = tune_all(n_trials=tune_trials)

        # Save tuning results
        tuned_path = root / "models" / "tuned_params.json"
        with open(tuned_path, "w") as f:
            json.dump(tuning_results, f, indent=2)
        print(f"Saved tuning results to {tuned_path}")

        # Build models from tuned params
        models = {}
        for r in tuning_results:
            try:
                models[r["model"]] = _build_model_from_params(r["model"], r["best_params"])
            except (ValueError, TypeError):
                pass
    elif params_path:
        # Load previously saved tuned params
        with open(params_path) as f:
            tuning_results = json.load(f)
        print(f"\nLoading tuned params from {params_path}")
        models = {}
        for r in tuning_results:
            try:
                models[r["model"]] = _build_model_from_params(r["model"], r["best_params"])
            except (ValueError, TypeError):
                pass
    else:
        models = _default_models()

    best_model_name = None
    best_score = -1
    best_model = None

    # Evaluate models on validation set (simple time split within train set)
    val_split_idx = int(len(X_train) * 0.8)
    X_t, X_v = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_t, y_v = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]

    # Precompute sample weights for classes that need them
    sw_t = _compute_sample_weights(y_t)
    sw_train = _compute_sample_weights(y_train)

    # Track all model scores for ensemble selection
    model_scores = {}

    print("\nModel Evaluation (Macro F1 on validation set):")
    for name, model in models.items():
        if name in SAMPLE_WEIGHT_MODELS:
            model.fit(X_t, y_t, sample_weight=sw_t)
        else:
            model.fit(X_t, y_t)
        preds = model.predict(X_v)
        score = f1_score(y_v, preds, average='macro')
        model_scores[name] = score
        print(f"  {name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = model

    # --- Soft-voting ensemble ---
    use_ensemble = False
    if ensemble and len(model_scores) >= 2:
        # Collect models within 0.05 macro F1 of the best individual
        threshold = best_score - 0.05
        top_model_names = [
            name for name, score in model_scores.items()
            if score >= threshold
        ]

        if len(top_model_names) >= 2:
            print(f"\nBuilding soft-voting ensemble from {len(top_model_names)} "
                  f"top models (within 0.05 of best {best_score:.4f})...")

            # Retrain each top model on the full sub-training set (X_t, y_t)
            # to get fresh fitted instances for the ensemble
            ensemble_estimators = []
            for name in top_model_names:
                # Build a fresh instance with the same config
                fresh_model = _default_models().get(name)
                if fresh_model is None:
                    continue
                if name in SAMPLE_WEIGHT_MODELS:
                    fresh_model.fit(X_t, y_t, sample_weight=sw_t)
                else:
                    fresh_model.fit(X_t, y_t)
                ensemble_estimators.append((name, fresh_model))

            if len(ensemble_estimators) >= 2:
                ens = SoftVotingEnsemble(ensemble_estimators)
                ens_preds = ens.predict(X_v)
                ens_score = f1_score(y_v, ens_preds, average='macro')

                ens_names = " + ".join(n for n, _ in ensemble_estimators)
                print(f"  Ensemble ({ens_names}): {ens_score:.4f} "
                      f"vs Best Individual ({best_model_name}): {best_score:.4f}")

                if ens_score >= best_score:
                    print(f"  -> Using ensemble (beats or ties best individual)")
                    best_model_name = f"Ensemble({ens_names})"
                    best_score = ens_score
                    best_model = ens
                    use_ensemble = True
                else:
                    print(f"  -> Keeping best individual ({best_model_name})")
        else:
            print(f"\nSkipping ensemble: only {len(top_model_names)} model(s) "
                  "within 0.05 of best score (need at least 2).")
    elif ensemble:
        print("\nSkipping ensemble: fewer than 2 models available.")

    print(f"\nBest Model: {best_model_name} (Retraining on full training set...)")

    # Retrain on full training set
    if use_ensemble:
        # Retrain each sub-model on the full training data
        retrained_estimators = []
        for name, _ in best_model.estimators:
            fresh = _default_models().get(name)
            if fresh is None:
                continue
            if name in SAMPLE_WEIGHT_MODELS:
                fresh.fit(X_train, y_train, sample_weight=sw_train)
            else:
                fresh.fit(X_train, y_train)
            retrained_estimators.append((name, fresh))
        best_model = SoftVotingEnsemble(retrained_estimators)
    else:
        if best_model_name in SAMPLE_WEIGHT_MODELS:
            best_model.fit(X_train, y_train, sample_weight=sw_train)
        else:
            best_model.fit(X_train, y_train)

    # Probability calibration
    # Skip calibration for ensembles — the ensemble already averages probabilities
    # from diverse model types, which self-calibrates, and CalibratedClassifierCV
    # doesn't support our custom SoftVotingEnsemble.
    if use_ensemble:
        print("Skipping probability calibration (ensemble already averages diverse models).")
    elif calibrate:
        cal_size = len(X_train)
        if cal_size < 500:
            method = 'sigmoid'
            print(f"Calibrating probabilities (sigmoid, cv=3) — dataset small ({cal_size} samples)...")
        else:
            method = 'isotonic'
            print(f"Calibrating probabilities (isotonic, cv=3)...")

        calibrated = CalibratedClassifierCV(
            estimator=best_model,
            method=method,
            cv=3,
        )
        # Pass sample_weight to calibration for models that use it.
        # Pipeline-wrapped models (LR) already handle class imbalance via
        # class_weight='balanced' and don't accept sample_weight through Pipeline.
        if best_model_name in SAMPLE_WEIGHT_MODELS:
            calibrated.fit(X_train, y_train, sample_weight=sw_train)
        else:
            calibrated.fit(X_train, y_train)
        best_model = calibrated
    else:
        print("Skipping probability calibration (--no-calibrate).")

    # --- Per-class threshold optimization ---
    # Use the validation split (X_v, y_v) to find thresholds that maximize macro F1.
    # We re-fit the validation split from the *full* training data so thresholds
    # are tuned against data the calibrated model was trained on but held out.
    print("\nOptimizing per-class decision thresholds on validation set...")
    best_thresholds, threshold_f1 = optimize_thresholds(best_model, X_v, y_v)

    baseline_preds = best_model.predict(X_v)
    from sklearn.metrics import f1_score as _f1
    baseline_f1 = _f1(y_v, baseline_preds, average='macro')
    print(f"  Baseline macro F1 (argmax):      {baseline_f1:.4f}")
    print(f"  Threshold-tuned macro F1:         {threshold_f1:.4f}")
    print(f"  Optimal thresholds: { {int(k): round(v, 2) for k, v in best_thresholds.items()} }")

    # Save the model and feature columns
    out_dir = root / "models"
    os.makedirs(out_dir, exist_ok=True)

    model_path = out_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    cols_path = out_dir / "feature_columns.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)

    # Save thresholds
    thresholds_path = out_dir / "thresholds.json"
    serializable = {str(int(k)): float(v) for k, v in best_thresholds.items()}
    with open(thresholds_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Feature columns saved to {cols_path}")
    print(f"Thresholds saved to {thresholds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and select best model.")
    parser.add_argument(
        "--no-calibrate", action="store_true", default=False,
        help="Skip probability calibration.",
    )
    parser.add_argument(
        "--tune", action="store_true", default=False,
        help="Run Optuna hyperparameter tuning before training.",
    )
    parser.add_argument(
        "--tune-trials", type=int, default=30,
        help="Number of Optuna trials per model (default: 30).",
    )
    parser.add_argument(
        "--params", type=str, default=None,
        help="Path to tuned_params.json to use instead of defaults.",
    )
    parser.add_argument(
        "--ensemble", action="store_true", default=False,
        help="Build a soft-voting ensemble from top models.",
    )
    args = parser.parse_args()
    train_and_select_model(
        calibrate=not args.no_calibrate,
        tune=args.tune,
        tune_trials=args.tune_trials,
        params_path=args.params,
        ensemble=args.ensemble,
    )
