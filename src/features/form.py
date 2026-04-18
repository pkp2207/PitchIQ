import pandas as pd
import numpy as np

def compute_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling form features (last 5 and 10 matches) for each team."""
    
    # Transform data into team-match level to compute form easily
    home_df = df[['date', 'home_team', 'home_score', 'away_score', 'outcome']].copy()
    home_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded', 'outcome']
    home_df['is_home'] = 1
    
    away_df = df[['date', 'away_team', 'away_score', 'home_score', 'outcome']].copy()
    away_df.columns = ['date', 'team', 'goals_scored', 'goals_conceded', 'outcome']
    # Reverse outcome for away team
    away_df['outcome'] = away_df['outcome'] * -1
    away_df['is_home'] = 0
    
    team_matches = pd.concat([home_df, away_df]).sort_values(by=['team', 'date']).reset_index(drop=True)
    
    # Win, draw, loss flags
    team_matches['win'] = (team_matches['outcome'] == 1).astype(int)
    team_matches['draw'] = (team_matches['outcome'] == 0).astype(int)
    team_matches['loss'] = (team_matches['outcome'] == -1).astype(int)
    
    # Compute rolling stats (shift by 1 to only use PAST matches)
    for window in [5, 10]:
        team_matches[f'win_rate_{window}'] = team_matches.groupby('team')['win'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        team_matches[f'goals_scored_avg_{window}'] = team_matches.groupby('team')['goals_scored'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        team_matches[f'goals_conceded_avg_{window}'] = team_matches.groupby('team')['goals_conceded'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    
    # Fill NAs for teams with no prior matches
    team_matches = team_matches.fillna(0)
    
    # Merge back to the main dataframe
    # First home features
    home_features = team_matches[team_matches['is_home'] == 1].copy()
    home_features = home_features[['date', 'team'] + [c for c in team_matches.columns if 'rate_' in c or 'avg_' in c]]
    home_features.columns = ['date', 'home_team'] + [f'home_{c}' for c in home_features.columns if c not in ['date', 'team']]
    
    df = df.merge(home_features, on=['date', 'home_team'], how='left')
    
    # Then away features
    away_features = team_matches[team_matches['is_home'] == 0].copy()
    away_features = away_features[['date', 'team'] + [c for c in team_matches.columns if 'rate_' in c or 'avg_' in c]]
    away_features.columns = ['date', 'away_team'] + [f'away_{c}' for c in away_features.columns if c not in ['date', 'team']]
    
    df = df.merge(away_features, on=['date', 'away_team'], how='left')
    
    return df
