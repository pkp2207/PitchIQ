import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

    def fit(self, X, y):
        self.xgb.fit(X, _encode_labels(y))
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


def train_and_select_model():
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

    # Initialize models
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
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

    best_model_name = None
    best_score = -1
    best_model = None

    # Evaluate models on validation set (simple time split within train set)
    val_split_idx = int(len(X_train) * 0.8)
    X_t, X_v = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_t, y_v = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]

    print("\nModel Evaluation (Macro F1 on validation set):")
    for name, model in models.items():
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        score = f1_score(y_v, preds, average='macro')
        print(f"  {name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = model

    print(f"\nBest Model: {best_model_name} (Retraining on full training set...)")

    # Retrain best model on full training set
    best_model.fit(X_train, y_train)

    # Save the model and feature columns
    out_dir = root / "models"
    os.makedirs(out_dir, exist_ok=True)

    model_path = out_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    cols_path = out_dir / "feature_columns.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)

    print(f"Model saved to {model_path}")
    print(f"Feature columns saved to {cols_path}")


if __name__ == "__main__":
    train_and_select_model()
