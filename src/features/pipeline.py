import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd
from pathlib import Path
from src.data.loader import get_project_root
from src.data.cleaner import clean_and_prepare_matches
from src.features.elo import compute_elo_features
from src.features.form import compute_team_form
from src.features.head_to_head import compute_h2h_features

def run_feature_pipeline(cutoff_year: int = 1990) -> pd.DataFrame:
    print("Loading and cleaning data...")
    df = clean_and_prepare_matches(cutoff_year)
    
    print("Computing Elo features...")
    df = compute_elo_features(df)
    
    print("Computing form features...")
    df = compute_team_form(df)
    
    print("Computing H2H features...")
    df = compute_h2h_features(df)
    
    # Process additional features
    df['neutral'] = df['neutral'].astype(int)
    
    # Friendly vs Competitive match
    df['is_friendly'] = (df['tournament'] == 'Friendly').astype(int)
    
    # Select final columns to save
    feature_cols = [
        'date', 'home_team', 'away_team', 'tournament', 'neutral', 'is_friendly',
        'home_elo_before', 'away_elo_before', 'elo_diff',
        'home_win_rate_5', 'home_win_rate_10', 'home_goals_scored_avg_5', 'home_goals_scored_avg_10',
        'home_goals_conceded_avg_5', 'home_goals_conceded_avg_10',
        'away_win_rate_5', 'away_win_rate_10', 'away_goals_scored_avg_5', 'away_goals_scored_avg_10',
        'away_goals_conceded_avg_5', 'away_goals_conceded_avg_10',
        'h2h_home_win_rate', 'h2h_avg_goal_diff', 'h2h_num_meetings',
        'outcome' # Target variable
    ]
    
    # Keep only relevant columns and drop rows with NAs that might have been introduced
    df_features = df[feature_cols].copy()
    df_features = df_features.fillna(0)
    
    return df_features

if __name__ == "__main__":
    print("Running feature engineering pipeline...")
    df_features = run_feature_pipeline(cutoff_year=1990)
    
    out_dir = get_project_root() / "data" / "processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "features.csv"
    
    df_features.to_csv(out_path, index=False)
    print(f"Saved features to {out_path} ({len(df_features)} matches)")
