import json
import logging

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    log_loss, brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize
from src.data.loader import get_project_root
from src.models.threshold import apply_thresholds

logger = logging.getLogger(__name__)


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

    # --- Raw (argmax) predictions ---
    raw_preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    classes = model.classes_

    raw_acc = accuracy_score(y_test, raw_preds)
    raw_f1 = f1_score(y_test, raw_preds, average='macro')
    raw_cm = confusion_matrix(y_test, raw_preds)
    raw_report = classification_report(y_test, raw_preds, output_dict=True)

    # --- Threshold-adjusted predictions ---
    thresholds_path = root / "models" / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, 'r') as f:
            raw_thresh = json.load(f)
        thresholds = {int(k): float(v) for k, v in raw_thresh.items()}
    else:
        thresholds = {-1: 1.0, 0: 1.0, 1: 1.0}

    adj_probs, adj_preds = apply_thresholds(probs, thresholds, classes)

    adj_acc = accuracy_score(y_test, adj_preds)
    adj_f1 = f1_score(y_test, adj_preds, average='macro')
    adj_cm = confusion_matrix(y_test, adj_preds)
    adj_report = classification_report(y_test, adj_preds, output_dict=True)

    # Probability calibration metrics (computed on raw probs, not adjusted)
    logloss = log_loss(y_test, probs, labels=classes)

    # Brier score: computed per-class then averaged (multiclass extension)
    y_bin = label_binarize(y_test, classes=classes)
    brier_per_class = []
    for i in range(len(classes)):
        bs = brier_score_loss(y_bin[:, i], probs[:, i])
        brier_per_class.append(bs)
    brier_avg = float(np.mean(brier_per_class))

    # Detect whether the model is probability-calibrated
    is_calibrated = isinstance(model, CalibratedClassifierCV)

    # Reliability diagram data (per-class calibration curves)
    reliability = {}
    for i, cls in enumerate(classes):
        try:
            fraction_pos, mean_predicted = calibration_curve(
                y_bin[:, i], probs[:, i], n_bins=10, strategy='uniform',
            )
            reliability[str(int(cls))] = {
                'fraction_of_positives': fraction_pos.tolist(),
                'mean_predicted_value': mean_predicted.tolist(),
            }
        except Exception:
            # Not enough data in some bins — skip this class
            pass

    results = {
        'Raw (argmax)': {
            'Accuracy': raw_acc,
            'Macro F1': raw_f1,
            'Confusion Matrix': raw_cm.tolist(),
            'Classification Report': raw_report,
        },
        'Threshold-adjusted': {
            'Accuracy': adj_acc,
            'Macro F1': adj_f1,
            'Thresholds': {str(k): v for k, v in thresholds.items()},
            'Confusion Matrix': adj_cm.tolist(),
            'Classification Report': adj_report,
        },
        'Log Loss': logloss,
        'Brier Score (avg)': brier_avg,
        'Calibrated': is_calibrated,
        'Reliability Diagram': reliability,
    }

    # --- Log results ---
    logger.info("=" * 60)
    logger.info("Evaluation Results on Test Set (Last 10% of time)")
    logger.info("=" * 60)

    logger.info("--- Raw predictions (argmax) ---")
    logger.info(f"  Accuracy:    {raw_acc:.4f}")
    logger.info(f"  Macro F1:    {raw_f1:.4f}")
    logger.info("  Confusion Matrix:")
    logger.info(f"  {raw_cm}")
    logger.info(f"  Per-class report:")
    for cls_key in ['-1', '0', '1']:
        if cls_key in raw_report:
            r = raw_report[cls_key]
            logger.info(f"    Class {cls_key:>2}: precision={r['precision']:.3f}  "
                        f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")

    logger.info("--- Threshold-adjusted predictions ---")
    logger.info(f"  Thresholds:  { {int(k): round(v, 2) for k, v in thresholds.items()} }")
    logger.info(f"  Accuracy:    {adj_acc:.4f}")
    logger.info(f"  Macro F1:    {adj_f1:.4f}")
    logger.info("  Confusion Matrix:")
    logger.info(f"  {adj_cm}")
    logger.info(f"  Per-class report:")
    for cls_key in ['-1', '0', '1']:
        if cls_key in adj_report:
            r = adj_report[cls_key]
            logger.info(f"    Class {cls_key:>2}: precision={r['precision']:.3f}  "
                        f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")

    delta_f1 = adj_f1 - raw_f1
    direction = "+" if delta_f1 >= 0 else ""
    logger.info(f"  Macro F1 change: {direction}{delta_f1:.4f}")

    logger.info("--- Calibration metrics ---")
    logger.info(f"  Log Loss:    {logloss:.4f}")
    logger.info(f"  Brier Score: {brier_avg:.4f}")
    logger.info(f"  Calibrated:  {is_calibrated}")

    out_path = root / "models" / "evaluation_report.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved report to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    evaluate_model()
