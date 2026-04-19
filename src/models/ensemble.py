"""Soft-voting ensemble that averages predict_proba outputs from multiple fitted models."""

import numpy as np


class SoftVotingEnsemble:
    """A lightweight soft-voting ensemble that combines fitted models.

    Unlike sklearn's VotingClassifier, this class:
    - Takes *already fitted* models (no re-fitting required)
    - Avoids the sample_weight / Pipeline compatibility issues
    - Provides averaged feature_importances_ for tree-based sub-models

    Parameters
    ----------
    estimators : list of (name, fitted_model) tuples
        Each model must support predict_proba() and have a classes_ attribute.
    """

    def __init__(self, estimators):
        if not estimators:
            raise ValueError("estimators list must not be empty")
        self.estimators = estimators
        self.estimator_names = [name for name, _ in estimators]
        self.estimators_ = [model for _, model in estimators]
        # All models must agree on the class labels
        self.classes_ = self.estimators_[0].classes_

    def predict_proba(self, X):
        """Average the predicted probabilities from all sub-models."""
        probas = np.array([m.predict_proba(X) for m in self.estimators_])
        return np.mean(probas, axis=0)

    def predict(self, X):
        """Return the class with the highest averaged probability."""
        avg_proba = self.predict_proba(X)
        return self.classes_[np.argmax(avg_proba, axis=1)]

    @property
    def feature_importances_(self):
        """Average feature_importances_ across sub-models that expose them.

        Models without feature_importances_ (e.g., LogisticRegression) are
        skipped.  Returns None if no sub-model has importances.
        """
        importances_list = []
        for model in self.estimators_:
            # Unwrap common wrappers to find the raw estimator
            raw = model
            if hasattr(raw, 'xgb'):
                raw = raw.xgb
            if hasattr(raw, 'feature_importances_'):
                importances_list.append(raw.feature_importances_)
        if not importances_list:
            return None
        return np.mean(importances_list, axis=0)

    def __repr__(self):
        names = " + ".join(self.estimator_names)
        return f"SoftVotingEnsemble({names})"
