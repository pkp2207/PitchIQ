import pandas as pd
import numpy as np

def compute_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling form features (last 5 and 10 matches) for each team."""

    # Add a unique match index to prevent merge inflation when a team
    # plays multiple matches on the same date.
    df = df.reset_index(drop=True)
    df['_match_idx'] = df.index

    # Transform data into team-match level to compute form easily
    home_df = df[['_match_idx', 'date', 'home_team', 'home_score', 'away_score', 'outcome']].copy()
    home_df.columns = ['_match_idx', 'date', 'team', 'goals_scored', 'goals_conceded', 'outcome']
    home_df['is_home'] = 1

    away_df = df[['_match_idx', 'date', 'away_team', 'away_score', 'home_score', 'outcome']].copy()
    away_df.columns = ['_match_idx', 'date', 'team', 'goals_scored', 'goals_conceded', 'outcome']
    # Reverse outcome for away team
    away_df['outcome'] = away_df['outcome'] * -1
    away_df['is_home'] = 0

    team_matches = pd.concat([home_df, away_df]).sort_values(by=['team', 'date']).reset_index(drop=True)

    # Win, draw, loss flags
    team_matches['win'] = (team_matches['outcome'] == 1).astype(int)
    team_matches['draw'] = (team_matches['outcome'] == 0).astype(int)
    team_matches['loss'] = (team_matches['outcome'] == -1).astype(int)

    # Goal difference per team-match (goals_scored - goals_conceded)
    team_matches['goal_diff'] = team_matches['goals_scored'] - team_matches['goals_conceded']

    # Compute rolling stats (shift by 1 to only use PAST matches)
    form_cols = []
    for window in [5, 10]:
        col_wr = f'win_rate_{window}'
        col_gs = f'goals_scored_avg_{window}'
        col_gc = f'goals_conceded_avg_{window}'
        col_gd = f'goal_diff_avg_{window}'
        team_matches[col_wr] = team_matches.groupby('team')['win'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        team_matches[col_gs] = team_matches.groupby('team')['goals_scored'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        team_matches[col_gc] = team_matches.groupby('team')['goals_conceded'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        team_matches[col_gd] = team_matches.groupby('team')['goal_diff'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        form_cols.extend([col_wr, col_gs, col_gc, col_gd])

    # Home advantage: rolling win rate over the team's last 10 home matches
    # Only considers matches where the team was at home
    home_only = team_matches[team_matches['is_home'] == 1].copy()
    home_only['home_advantage'] = home_only.groupby('team')['win'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    # Merge home_advantage back into team_matches for home rows
    team_matches = team_matches.merge(
        home_only[['_match_idx', 'team', 'home_advantage']],
        on=['_match_idx', 'team'],
        how='left',
    )
    # Away rows get NaN for home_advantage; that's fine — we only use it for home team

    # Fill NAs for teams with no prior matches
    team_matches = team_matches.fillna(0)

    # Merge back using _match_idx for an exact 1:1 join
    home_features = team_matches[team_matches['is_home'] == 1][['_match_idx'] + form_cols + ['home_advantage']].copy()
    home_features.columns = ['_match_idx'] + [f'home_{c}' for c in form_cols] + ['home_advantage']
    df = df.merge(home_features, on='_match_idx', how='left')

    away_features = team_matches[team_matches['is_home'] == 0][['_match_idx'] + form_cols].copy()
    away_features.columns = ['_match_idx'] + [f'away_{c}' for c in form_cols]
    df = df.merge(away_features, on='_match_idx', how='left')

    df = df.drop(columns=['_match_idx'])
    return df
