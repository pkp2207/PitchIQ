import logging
import os
import pandas as pd
from pathlib import Path
from src.data.loader import get_project_root, load_players, load_sentiment
from src.data.cleaner import clean_and_prepare_matches
from src.features.elo import compute_elo_features
from src.features.form import compute_team_form
from src.features.head_to_head import compute_h2h_features
from src.features.streak import compute_streak_features
from src.features.squad_strength import compute_squad_features
from src.features.sentiment import compute_sentiment_features

logger = logging.getLogger(__name__)

def run_feature_pipeline(cutoff_year: int = 1990) -> pd.DataFrame:
    logger.info("Loading and cleaning data...")
    df = clean_and_prepare_matches(cutoff_year)

    logger.info("Computing Elo features...")
    df = compute_elo_features(df)

    logger.info("Computing form features...")
    df = compute_team_form(df)

    logger.info("Computing H2H features...")
    df = compute_h2h_features(df)

    logger.info("Computing streak and rest features...")
    df = compute_streak_features(df)

    # Phase 2: Squad strength features from Transfermarkt player data
    try:
        players_df = load_players()
        logger.info("Computing squad strength features...")
        df = compute_squad_features(df, players_df)
    except FileNotFoundError:
        logger.warning("Player data not found — skipping squad strength features. "
                       "Run: python scripts/generate_sample_players.py")
        # Add placeholder columns so the rest of the pipeline doesn't break
        df['home_squad_value'] = 0.0
        df['away_squad_value'] = 0.0
        df['squad_value_diff'] = 0.0
        df['home_avg_age'] = 0.0
        df['away_avg_age'] = 0.0

    # Phase 3: Sentiment features from news/social media data
    try:
        sentiment_df = load_sentiment()
        logger.info("Computing sentiment features...")
        df = compute_sentiment_features(df, sentiment_df)
    except FileNotFoundError:
        logger.warning("Sentiment data not found — skipping sentiment features. "
                       "Run: python scripts/generate_sample_sentiment.py")
        # Add placeholder columns so the rest of the pipeline doesn't break
        df['home_sentiment_avg'] = 0.0
        df['away_sentiment_avg'] = 0.0
        df['sentiment_diff'] = 0.0
        df['home_sentiment_volume'] = 0
        df['away_sentiment_volume'] = 0

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
        'home_goal_diff_avg_5', 'home_goal_diff_avg_10',
        'away_win_rate_5', 'away_win_rate_10', 'away_goals_scored_avg_5', 'away_goals_scored_avg_10',
        'away_goals_conceded_avg_5', 'away_goals_conceded_avg_10',
        'away_goal_diff_avg_5', 'away_goal_diff_avg_10',
        'home_advantage',
        'h2h_home_win_rate', 'h2h_avg_goal_diff', 'h2h_num_meetings',
        'home_streak', 'away_streak',
        'home_days_since_last', 'away_days_since_last',
        'home_squad_value', 'away_squad_value', 'squad_value_diff',
        'home_avg_age', 'away_avg_age',
        'home_sentiment_avg', 'away_sentiment_avg', 'sentiment_diff',
        'home_sentiment_volume', 'away_sentiment_volume',
        'outcome' # Target variable
    ]
    
    # Keep only relevant columns and drop rows with NAs that might have been introduced
    df_features = df[feature_cols].copy()
    df_features = df_features.fillna(0)
    
    return df_features

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Running feature engineering pipeline...")
    df_features = run_feature_pipeline(cutoff_year=1990)

    out_dir = get_project_root() / "data" / "processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "features.csv"

    df_features.to_csv(out_path, index=False)
    logger.info(f"Saved features to {out_path} ({len(df_features)} matches)")
