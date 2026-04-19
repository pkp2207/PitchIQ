import pandas as pd
import numpy as np


def compute_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes win/loss streak and days-since-last-match for each team.

    For each match, produces:
      - home_streak / away_streak: positive = consecutive wins, negative =
        consecutive losses, 0 = last result was a draw.  Reflects state
        BEFORE the current match (shift-by-1 semantics).
      - home_days_since_last / away_days_since_last: calendar days since the
        team last played any match (home or away).  First match defaults to 30.

    Iterates chronologically over a stacked (team-level) view of all matches
    to correctly track each team's history regardless of home/away role.
    """

    df = df.sort_values(by='date').reset_index(drop=True)
    df['_match_idx'] = df.index

    # ------------------------------------------------------------------ #
    # Stack home and away into a single team-level table
    # ------------------------------------------------------------------ #
    home_df = df[['_match_idx', 'date', 'home_team', 'outcome']].copy()
    home_df.columns = ['_match_idx', 'date', 'team', 'outcome']
    home_df['is_home'] = 1

    away_df = df[['_match_idx', 'date', 'away_team', 'outcome']].copy()
    away_df.columns = ['_match_idx', 'date', 'team', 'outcome']
    # Reverse outcome for the away team
    away_df['outcome'] = away_df['outcome'] * -1
    away_df['is_home'] = 0

    team_matches = pd.concat([home_df, away_df]).sort_values(
        by=['team', 'date']
    ).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Compute streak and days-since-last via chronological iteration
    # ------------------------------------------------------------------ #
    # We track per-team state and record the *previous* state for each row
    # (shift-by-1 semantics).
    team_state = {}  # team -> {'streak': int, 'last_date': pd.Timestamp}

    streaks = np.zeros(len(team_matches), dtype=int)
    days_since = np.full(len(team_matches), 30.0)

    for i, row in team_matches.iterrows():
        team = row['team']
        match_date = row['date']
        outcome = row['outcome']  # 1=win, 0=draw, -1=loss (from this team's perspective)

        if team in team_state:
            # Record the state BEFORE this match (shift-by-1)
            streaks[i] = team_state[team]['streak']
            delta = (match_date - team_state[team]['last_date']).days
            days_since[i] = max(delta, 0)
        else:
            # First match for this team — defaults
            streaks[i] = 0
            days_since[i] = 30

        # Update state AFTER this match
        if outcome == 1:
            # Win: if previous streak was positive, extend; otherwise start at 1
            prev = team_state.get(team, {}).get('streak', 0)
            new_streak = (prev + 1) if prev > 0 else 1
        elif outcome == -1:
            # Loss: if previous streak was negative, extend; otherwise start at -1
            prev = team_state.get(team, {}).get('streak', 0)
            new_streak = (prev - 1) if prev < 0 else -1
        else:
            new_streak = 0

        team_state[team] = {'streak': new_streak, 'last_date': match_date}

    team_matches['streak'] = streaks
    team_matches['days_since_last'] = days_since

    # ------------------------------------------------------------------ #
    # Merge back to match-level DataFrame
    # ------------------------------------------------------------------ #
    merge_cols = ['streak', 'days_since_last']

    home_feat = team_matches[team_matches['is_home'] == 1][
        ['_match_idx'] + merge_cols
    ].copy()
    home_feat.columns = ['_match_idx', 'home_streak', 'home_days_since_last']
    df = df.merge(home_feat, on='_match_idx', how='left')

    away_feat = team_matches[team_matches['is_home'] == 0][
        ['_match_idx'] + merge_cols
    ].copy()
    away_feat.columns = ['_match_idx', 'away_streak', 'away_days_since_last']
    df = df.merge(away_feat, on='_match_idx', how='left')

    df = df.drop(columns=['_match_idx'])
    return df
