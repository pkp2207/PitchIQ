import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from src.data.loader import get_project_root
from src.models.threshold import apply_thresholds


class MatchPredictor:
    def __init__(self):
        self.root = get_project_root()
        self.model_path = self.root / "models" / "best_model.pkl"
        self.features_path = self.root / "models" / "feature_columns.json"

        if not self.model_path.exists() or not self.features_path.exists():
            raise FileNotFoundError("Model or feature columns not found. Run src/models/train.py first.")

        self.model = joblib.load(self.model_path)
        with open(self.features_path, 'r') as f:
            self.feature_cols = json.load(f)

        # Load per-class decision thresholds (default to 1.0 if not present)
        thresholds_path = self.root / "models" / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                raw = json.load(f)
            self.thresholds = {int(k): float(v) for k, v in raw.items()}
        else:
            self.thresholds = {-1: 1.0, 0: 1.0, 1: 1.0}

        # Load historical features for constructing prediction inputs
        self.features_data_path = self.root / "data" / "processed" / "features.csv"
        if self.features_data_path.exists():
            self.df_features = pd.read_csv(self.features_data_path)
        else:
            self.df_features = None

        # Set up SHAP explainer
        self.explainer = None
        if HAS_SHAP:
            self._init_shap()

    def _get_underlying_model(self):
        """Unwrap Pipeline, CalibratedClassifierCV, SoftVotingEnsemble, or custom wrappers to get the raw estimator."""
        from sklearn.pipeline import Pipeline
        from sklearn.calibration import CalibratedClassifierCV
        from src.models.ensemble import SoftVotingEnsemble
        model = self.model

        # Unwrap CalibratedClassifierCV first — it wraps another model
        if isinstance(model, CalibratedClassifierCV):
            # For a fitted CalibratedClassifierCV, the per-fold fitted
            # estimators live in calibrated_classifiers_[i].estimator.
            # Use the first fold's fitted estimator so that
            # feature_importances_ / coef_ are available.
            if hasattr(model, 'calibrated_classifiers_') and model.calibrated_classifiers_:
                model = model.calibrated_classifiers_[0].estimator
            else:
                # Unfitted — fall back to the template estimator
                model = model.estimator

        # Unwrap SoftVotingEnsemble — extract first tree-based sub-model for SHAP,
        # or fall back to the ensemble itself (which has feature_importances_).
        if isinstance(model, SoftVotingEnsemble):
            # Prefer a tree-based sub-model for SHAP compatibility
            for sub_model in model.estimators_:
                raw = sub_model
                if hasattr(raw, 'xgb'):
                    return raw.xgb
                if hasattr(raw, 'feature_importances_'):
                    return raw
            # No tree-based model found — return ensemble itself
            # (its feature_importances_ property will be used as fallback)
            return model

        # Unwrap Pipeline
        if isinstance(model, Pipeline):
            model = model.named_steps['clf']

        # Our _XGBWrapper stores the real model in .xgb
        if hasattr(model, 'xgb'):
            return model.xgb

        return model

    def _init_shap(self):
        raw = self._get_underlying_model()
        try:
            if hasattr(raw, 'feature_importances_'):
                # Tree-based model
                self.explainer = shap.TreeExplainer(raw)
            elif hasattr(raw, 'coef_'):
                # Linear model — needs background data
                if self.df_features is not None:
                    bg = self.df_features[self.feature_cols].sample(
                        min(100, len(self.df_features)), random_state=42,
                    )
                    self.explainer = shap.LinearExplainer(raw, bg)
        except Exception:
            self.explainer = None

    def _get_shap_values(self, X_pred, predicted_class):
        """Return top 5 SHAP values (signed) for the predicted class."""
        if self.explainer is None:
            return None

        try:
            sv = self.explainer.shap_values(X_pred)
        except Exception:
            return None

        # sv can be a list of arrays (one per class) or a single array
        classes = np.array([-1, 0, 1])
        if isinstance(sv, list):
            class_idx = np.where(classes == predicted_class)[0][0]
            vals = sv[class_idx][0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            class_idx = np.where(classes == predicted_class)[0][0]
            vals = sv[0, :, class_idx]
        else:
            vals = sv[0] if sv.ndim == 2 else sv

        top_idx = np.argsort(np.abs(vals))[::-1][:5]
        return [
            {'feature': self.feature_cols[i], 'shap_value': float(vals[i])}
            for i in top_idx
        ]

    def _get_basic_importances(self, X_pred, predicted_class):
        """Fallback: use feature_importances_ or coef_."""
        raw = self._get_underlying_model()
        top_features = []
        if hasattr(raw, 'feature_importances_'):
            importances = raw.feature_importances_
            indices = np.argsort(importances)[::-1][:5]
            for idx in indices:
                top_features.append({
                    'feature': self.feature_cols[idx],
                    'shap_value': float(importances[idx]),
                })
        elif hasattr(raw, 'coef_'):
            classes = raw.classes_
            class_idx = np.where(classes == predicted_class)[0][0]
            coefs = raw.coef_[class_idx]
            indices = np.argsort(np.abs(coefs))[::-1][:5]
            for idx in indices:
                top_features.append({
                    'feature': self.feature_cols[idx],
                    'shap_value': float(coefs[idx]),
                })
        return top_features

    def _get_latest_team_stats(self, team: str) -> dict:
        """Get the latest known feature values for a team from any match they played.

        Scans the features dataframe for the most recent row where the team
        appeared as either home_team or away_team, then extracts feature values
        with canonical (unprefixed) keys.
        """
        if self.df_features is None:
            return {}

        # Find the most recent match this team played in ANY role
        mask = (self.df_features['home_team'] == team) | (self.df_features['away_team'] == team)
        team_rows = self.df_features[mask]
        if len(team_rows) == 0:
            return {}

        latest = team_rows.iloc[-1]
        was_home = (latest['home_team'] == team)
        prefix = 'home_' if was_home else 'away_'

        stats = {}
        # Elo
        stats['elo'] = latest[f'{prefix}elo_before']

        # Form features (win_rate, goals_scored_avg, goals_conceded_avg, goal_diff_avg)
        for window in [5, 10]:
            for feat in ['win_rate', 'goals_scored_avg', 'goals_conceded_avg', 'goal_diff_avg']:
                key = f'{feat}_{window}'
                stats[key] = latest.get(f'{prefix}{key}', 0)

        # Streak and rest
        stats['streak'] = latest.get(f'{prefix}streak', 0)
        stats['days_since_last'] = latest.get(f'{prefix}days_since_last', 30)

        # Squad strength
        stats['squad_value'] = latest.get(f'{prefix}squad_value', 0)
        stats['avg_age'] = latest.get(f'{prefix}avg_age', 0)

        # Sentiment
        stats['sentiment_avg'] = latest.get(f'{prefix}sentiment_avg', 0)
        stats['sentiment_volume'] = latest.get(f'{prefix}sentiment_volume', 0)

        # Home advantage — only from rows where team was home
        home_rows = self.df_features[self.df_features['home_team'] == team]
        if len(home_rows) > 0:
            stats['home_advantage'] = home_rows.iloc[-1].get('home_advantage', 0)
        else:
            stats['home_advantage'] = 0

        return stats

    def predict_match(self, home_team: str, away_team: str,
                      tournament_type: str = "Friendly", date: str = None):
        if self.df_features is None:
            raise ValueError("Features data not found. Run feature pipeline.")

        # Get latest stats for each team from their most recent appearance
        home_stats = self._get_latest_team_stats(home_team)
        away_stats = self._get_latest_team_stats(away_team)

        # Extract H2H stats
        h2h_matches = self.df_features[
            ((self.df_features['home_team'] == home_team) & (self.df_features['away_team'] == away_team)) |
            ((self.df_features['home_team'] == away_team) & (self.df_features['away_team'] == home_team))
        ]
        latest_h2h = h2h_matches.iloc[-1] if len(h2h_matches) > 0 else None

        # Construct feature vector
        match_features = {}
        for col in self.feature_cols:
            if col == 'neutral':
                match_features[col] = 0
            elif col == 'is_friendly':
                match_features[col] = 1 if tournament_type == 'Friendly' else 0
            elif col == 'home_elo_before':
                match_features[col] = home_stats.get('elo', 1500)
            elif col == 'away_elo_before':
                match_features[col] = away_stats.get('elo', 1500)
            elif col == 'elo_diff':
                match_features[col] = home_stats.get('elo', 1500) - away_stats.get('elo', 1500)
            elif col == 'home_advantage':
                match_features[col] = home_stats.get('home_advantage', 0)
            elif col == 'squad_value_diff':
                match_features[col] = home_stats.get('squad_value', 0) - away_stats.get('squad_value', 0)
            elif col == 'sentiment_diff':
                match_features[col] = home_stats.get('sentiment_avg', 0) - away_stats.get('sentiment_avg', 0)
            elif col.startswith('home_'):
                # Map home_win_rate_5 -> win_rate_5 in home_stats
                suffix = col[len('home_'):]
                match_features[col] = home_stats.get(suffix, 0)
            elif col.startswith('away_'):
                # Map away_win_rate_5 -> win_rate_5 in away_stats
                suffix = col[len('away_'):]
                match_features[col] = away_stats.get(suffix, 0)
            elif col.startswith('h2h_'):
                if latest_h2h is not None:
                    if latest_h2h['home_team'] == home_team:
                        match_features[col] = latest_h2h[col]
                    else:
                        if col == 'h2h_home_win_rate':
                            match_features[col] = 1 - latest_h2h[col]
                        elif col == 'h2h_avg_goal_diff':
                            match_features[col] = -latest_h2h[col]
                        else:
                            match_features[col] = latest_h2h[col]
                else:
                    match_features[col] = 0
            else:
                match_features[col] = 0

        X_pred = pd.DataFrame([match_features])[self.feature_cols]

        # For Pipeline models we need to transform X for SHAP but predict with the pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.calibration import CalibratedClassifierCV
        from src.models.ensemble import SoftVotingEnsemble

        # Find the inner pipeline (if any) for SHAP data scaling
        inner_model = self.model
        if isinstance(inner_model, CalibratedClassifierCV):
            if hasattr(inner_model, 'calibrated_classifiers_') and inner_model.calibrated_classifiers_:
                inner_model = inner_model.calibrated_classifiers_[0].estimator
            else:
                inner_model = inner_model.estimator
        # SoftVotingEnsemble doesn't wrap a Pipeline, so skip unwrapping
        if isinstance(inner_model, SoftVotingEnsemble):
            inner_model = None  # No scaling needed

        raw_probs = self.model.predict_proba(X_pred)
        classes = self.model.classes_

        # Apply per-class threshold scaling to boost under-predicted classes
        adjusted_probs, adjusted_preds = apply_thresholds(
            raw_probs, self.thresholds, classes,
        )
        probs = adjusted_probs[0]
        predicted_class = adjusted_preds[0]

        if inner_model is not None and isinstance(inner_model, Pipeline):
            # SHAP needs the scaled data
            X_shap = pd.DataFrame(
                inner_model.named_steps['scaler'].transform(X_pred),
                columns=self.feature_cols,
            )
        else:
            X_shap = X_pred

        prob_dict = {}
        for cls, prob in zip(classes, probs):
            if cls == 1:
                prob_dict['Win'] = float(prob)
            elif cls == 0:
                prob_dict['Draw'] = float(prob)
            elif cls == -1:
                prob_dict['Loss'] = float(prob)

        outcome_map = {1: 'Win', 0: 'Draw', -1: 'Loss'}

        # Get SHAP explanations (or fallback to basic importances)
        top_features = self._get_shap_values(X_shap, predicted_class)
        if top_features is None:
            top_features = self._get_basic_importances(X_shap, predicted_class)

        return {
            'outcome': outcome_map[predicted_class],
            'probabilities': prob_dict,
            'top_features': top_features,
        }

    def get_h2h_history(self, home_team: str, away_team: str, last_n: int = 5):
        """Return the last N head-to-head encounters between the two teams."""
        if self.df_features is None:
            return []

        mask = (
            ((self.df_features['home_team'] == home_team) & (self.df_features['away_team'] == away_team)) |
            ((self.df_features['home_team'] == away_team) & (self.df_features['away_team'] == home_team))
        )
        h2h = self.df_features[mask].tail(last_n)

        records = []
        for _, row in h2h.iterrows():
            records.append({
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'outcome': int(row['outcome']),
            })
        return records
