import os
import pandas as pd
from pathlib import Path
from src.data.loader import load_results, get_project_root

def clean_and_prepare_matches(cutoff_year: int = 1990) -> pd.DataFrame:
    """
    Cleans the raw match dataset and prepares it for feature engineering.
    """
    df = load_results()
    
    # Filter by cutoff year
    df = df[df['date'].dt.year >= cutoff_year].copy()
    
    # Drop rows with missing scores
    df = df.dropna(subset=['home_score', 'away_score'])
    
    # Standardize types
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    # Encode Target (Win = 1, Draw = 0, Loss = -1) from home team's perspective
    def get_outcome(row):
        if row['home_score'] > row['away_score']:
            return 1
        elif row['home_score'] == row['away_score']:
            return 0
        else:
            return -1
            
    df['outcome'] = df.apply(get_outcome, axis=1)
    
    # Normalize the 'neutral' column to boolean-like int
    if 'neutral' in df.columns:
        df['neutral'] = df['neutral'].map(
            lambda v: 1 if str(v).strip().upper() in ('TRUE', '1') else 0
        )

    # Sort chronologically (very important for rolling features)
    df = df.sort_values(by='date').reset_index(drop=True)

    return df

if __name__ == "__main__":
    print("Cleaning matches data (cutoff year 1990)...")
    df_cleaned = clean_and_prepare_matches(cutoff_year=1990)
    
    out_dir = get_project_root() / "data" / "processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "matches_cleaned.csv"
    
    df_cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path} ({len(df_cleaned)} matches)")
