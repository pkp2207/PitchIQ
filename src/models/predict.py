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
        """Unwrap Pipeline or custom wrappers to get the raw estimator."""
        from sklearn.pipeline import Pipeline
        model = self.model
        if isinstance(model, Pipeline):
            return model.named_steps['clf']
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

    def predict_match(self, home_team: str, away_team: str,
                      tournament_type: str = "Friendly", date: str = None):
        if self.df_features is None:
            raise ValueError("Features data not found. Run feature pipeline.")

        # Extract latest stats for home team
        home_rows = self.df_features[self.df_features['home_team'] == home_team]
        home_stats = home_rows.iloc[-1] if len(home_rows) > 0 else None

        # Extract latest stats for away team
        away_rows = self.df_features[self.df_features['away_team'] == away_team]
        away_stats = away_rows.iloc[-1] if len(away_rows) > 0 else None

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
            elif col.startswith('home_') and not col.startswith('home_elo'):
                match_features[col] = home_stats[col] if home_stats is not None else 0
            elif col.startswith('away_') and not col.startswith('away_elo'):
                match_features[col] = away_stats[col] if away_stats is not None else 0
            elif col == 'home_elo_before':
                match_features[col] = home_stats['home_elo_before'] if home_stats is not None else 1500
            elif col == 'away_elo_before':
                match_features[col] = away_stats['away_elo_before'] if away_stats is not None else 1500
            elif col == 'elo_diff':
                match_features[col] = match_features.get('home_elo_before', 1500) - match_features.get('away_elo_before', 1500)
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
        if isinstance(self.model, Pipeline):
            probs = self.model.predict_proba(X_pred)[0]
            predicted_class = self.model.predict(X_pred)[0]
            # SHAP needs the scaled data
            X_shap = pd.DataFrame(
                self.model.named_steps['scaler'].transform(X_pred),
                columns=self.feature_cols,
            )
        else:
            probs = self.model.predict_proba(X_pred)[0]
            predicted_class = self.model.predict(X_pred)[0]
            X_shap = X_pred

        classes = self.model.classes_
        prob_dict = {}
        for cls, prob in zip(classes, probs):
            if cls == 1:
                prob_dict['Win'] = prob
            elif cls == 0:
                prob_dict['Draw'] = prob
            elif cls == -1:
                prob_dict['Loss'] = prob

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
