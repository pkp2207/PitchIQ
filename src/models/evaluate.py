import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    log_loss, brier_score_loss,
)
from sklearn.preprocessing import label_binarize
from src.data.loader import get_project_root


def evaluate_model():
    root = get_project_root()
    data_path = root / "data" / "processed" / "features.csv"
    model_path = root / "models" / "best_model.pkl"
    features_path = root / "models" / "feature_columns.json"

    if not data_path.exists() or not model_path.exists():
        raise FileNotFoundError("Missing data or model files.")

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    with open(features_path, 'r') as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df['outcome']

    # Same split as training to evaluate on holdout set
    split_idx = int(len(df) * 0.9)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model = joblib.load(model_path)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    classes = model.classes_

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro')
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # Probability calibration metrics
    logloss = log_loss(y_test, probs, labels=classes)

    # Brier score: computed per-class then averaged (multiclass extension)
    y_bin = label_binarize(y_test, classes=classes)
    brier_per_class = []
    for i in range(len(classes)):
        bs = brier_score_loss(y_bin[:, i], probs[:, i])
        brier_per_class.append(bs)
    brier_avg = float(np.mean(brier_per_class))

    results = {
        'Accuracy': acc,
        'Macro F1': f1_macro,
        'Log Loss': logloss,
        'Brier Score (avg)': brier_avg,
        'Confusion Matrix': cm.tolist(),
        'Classification Report': report,
    }

    print("Evaluation Results on Test Set (Last 10% of time):")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {f1_macro:.4f}")
    print(f"  Log Loss:    {logloss:.4f}")
    print(f"  Brier Score: {brier_avg:.4f}")
    print("Confusion Matrix:")
    print(cm)

    out_path = root / "models" / "evaluation_report.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    evaluate_model()
