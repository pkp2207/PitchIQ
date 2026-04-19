"""Per-class decision threshold tuning to improve macro F1.

Instead of using argmax on raw predicted probabilities, we scale each
class's probability by a learned threshold/weight and then take the argmax.
This lets us boost under-predicted classes (especially draws).
"""

import numpy as np
from itertools import product
from sklearn.metrics import f1_score


def apply_thresholds(probs, thresholds, classes):
    """Apply per-class threshold scaling to probability matrix and return adjusted predictions.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes)
        Raw predicted probabilities from the model.
    thresholds : dict
        Mapping of class label -> scaling factor, e.g. {-1: 1.0, 0: 1.8, 1: 0.9}.
    classes : array-like
        Ordered class labels matching the columns of *probs*.

    Returns
    -------
    adjusted_probs : np.ndarray, shape (n_samples, n_classes)
        Re-normalized probabilities after scaling.
    predictions : np.ndarray, shape (n_samples,)
        Predicted class labels (argmax of adjusted_probs).
    """
    scale = np.array([thresholds.get(c, 1.0) for c in classes])
    adjusted = probs * scale  # broadcast (n, k) * (k,)

    # Re-normalize so rows sum to 1
    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid division by zero
    adjusted_probs = adjusted / row_sums

    best_idx = np.argmax(adjusted_probs, axis=1)
    predictions = np.array([classes[i] for i in best_idx])

    return adjusted_probs, predictions


def optimize_thresholds(model, X_val, y_val, classes=(-1, 0, 1)):
    """Find per-class probability thresholds that maximize macro F1.

    Uses a grid search over scaling factors for each class. The draw class
    (0) gets a wider search range because it is typically under-predicted.

    Parameters
    ----------
    model : fitted estimator
        Must implement ``predict_proba(X)``.
    X_val : DataFrame or array
        Validation features.
    y_val : Series or array
        True labels for validation set.
    classes : tuple
        Ordered class labels, default ``(-1, 0, 1)``.

    Returns
    -------
    best_thresholds : dict
        Mapping of class label -> optimal scaling factor.
    best_f1 : float
        The macro F1 score achieved with the optimal thresholds.
    """
    probs = model.predict_proba(X_val)
    model_classes = model.classes_
    y_true = np.asarray(y_val)

    # Define search grids per class
    # Loss (-1) and Win (1): [0.5, 1.5] step 0.1
    # Draw (0): [0.5, 3.0] step 0.1 (wider range since draws are under-predicted)
    grid = {}
    for c in classes:
        if c == 0:
            grid[c] = np.round(np.arange(0.5, 3.05, 0.1), 2)
        else:
            grid[c] = np.round(np.arange(0.5, 1.55, 0.1), 2)

    # Baseline: no threshold adjustment (all 1.0)
    baseline_preds = model.predict(X_val)
    baseline_f1 = f1_score(y_true, baseline_preds, average='macro')

    best_f1 = baseline_f1
    best_thresholds = {c: 1.0 for c in classes}

    # Grid search
    ordered_classes = list(classes)
    grid_values = [grid[c] for c in ordered_classes]

    total = 1
    for g in grid_values:
        total *= len(g)
    print(f"  Searching {total} threshold combinations...")

    for combo in product(*grid_values):
        thresholds = {c: v for c, v in zip(ordered_classes, combo)}
        _, preds = apply_thresholds(probs, thresholds, model_classes)
        score = f1_score(y_true, preds, average='macro')

        if score > best_f1:
            best_f1 = score
            best_thresholds = thresholds.copy()

    return best_thresholds, best_f1
