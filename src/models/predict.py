import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import get_project_root
from src.features.pipeline import run_feature_pipeline

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
            
        # We need historical features. For simplicity in prediction, 
        # we can load the latest features from features.csv
        self.features_data_path = self.root / "data" / "processed" / "features.csv"
        if self.features_data_path.exists():
            self.df_features = pd.read_csv(self.features_data_path)
        else:
            self.df_features = None

    def predict_match(self, home_team: str, away_team: str, tournament_type: str = "Friendly", date: str = None):
        """
        Predicts match outcome using historical averages for the teams.
        In a real scenario, we'd calculate features exactly up to the `date`.
        For simplicity, we grab the latest available features for each team.
        """
        if self.df_features is None:
            raise ValueError("Features data not found. Run feature pipeline.")
            
        # Extract latest stats for home team
        home_stats = self.df_features[self.df_features['home_team'] == home_team].iloc[-1] if len(self.df_features[self.df_features['home_team'] == home_team]) > 0 else None
        
        # Extract latest stats for away team
        away_stats = self.df_features[self.df_features['away_team'] == away_team].iloc[-1] if len(self.df_features[self.df_features['away_team'] == away_team]) > 0 else None
        
        # Extract H2H stats
        h2h_matches = self.df_features[((self.df_features['home_team'] == home_team) & (self.df_features['away_team'] == away_team)) | 
                                       ((self.df_features['home_team'] == away_team) & (self.df_features['away_team'] == home_team))]
        
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
                    # Determine perspective
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
                
        # Convert to dataframe in correct order
        X_pred = pd.DataFrame([match_features])[self.feature_cols]
        
        # Predict
        probs = self.model.predict_proba(X_pred)[0]
        classes = self.model.classes_
        
        # classes usually are [-1, 0, 1] for Loss, Draw, Win
        prob_dict = {}
        for cls, prob in zip(classes, probs):
            if cls == 1:
                prob_dict['Win'] = prob
            elif cls == 0:
                prob_dict['Draw'] = prob
            elif cls == -1:
                prob_dict['Loss'] = prob
                
        predicted_class = self.model.predict(X_pred)[0]
        outcome_map = {1: 'Win', 0: 'Draw', -1: 'Loss'}
        
        # Get top features (simple coefficient or feature importance)
        top_features = []
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:5]
            for idx in indices:
                top_features.append({
                    'feature': self.feature_cols[idx],
                    'importance': float(importances[idx])
                })
        elif hasattr(self.model, 'coef_'):
            # For logistic regression, get absolute coefficients for the predicted class
            class_idx = np.where(self.model.classes_ == predicted_class)[0][0]
            coefs = np.abs(self.model.coef_[class_idx])
            indices = np.argsort(coefs)[::-1][:5]
            for idx in indices:
                top_features.append({
                    'feature': self.feature_cols[idx],
                    'importance': float(self.model.coef_[class_idx][idx])
                })
                
        return {
            'outcome': outcome_map[predicted_class],
            'probabilities': prob_dict,
            'top_features': top_features
        }
