import pandas as pd
import numpy as np

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes historical head-to-head statistics prior to each match."""
    
    # We want to calculate the H2H stats for home_team vs away_team *before* each match date
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Create a unique matchup identifier independent of who is home/away
    df['matchup'] = df.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)
    
    h2h_home_win_rate = []
    h2h_avg_goal_diff = []
    h2h_num_meetings = []
    
    matchup_history = {}
    
    for _, row in df.iterrows():
        matchup = row['matchup']
        home_team = row['home_team']
        
        history = matchup_history.get(matchup, [])
        
        if not history:
            h2h_home_win_rate.append(0.0)
            h2h_avg_goal_diff.append(0.0)
            h2h_num_meetings.append(0)
        else:
            wins = 0
            goal_diff_sum = 0
            for prior_match in history:
                # prior_match is a dict: {'home_team': str, 'outcome': int, 'goal_diff': int}
                if prior_match['home_team'] == home_team:
                    if prior_match['outcome'] == 1:
                        wins += 1
                    goal_diff_sum += prior_match['goal_diff']
                else:
                    if prior_match['outcome'] == -1: # Away team won, which means our current home team won
                        wins += 1
                    goal_diff_sum -= prior_match['goal_diff'] # Reverse goal diff
                    
            h2h_home_win_rate.append(wins / len(history))
            h2h_avg_goal_diff.append(goal_diff_sum / len(history))
            h2h_num_meetings.append(len(history))
            
        # Update history
        goal_diff = row['home_score'] - row['away_score']
        matchup_history.setdefault(matchup, []).append({
            'home_team': row['home_team'],
            'outcome': row['outcome'],
            'goal_diff': goal_diff
        })
        
    df['h2h_home_win_rate'] = h2h_home_win_rate
    df['h2h_avg_goal_diff'] = h2h_avg_goal_diff
    df['h2h_num_meetings'] = h2h_num_meetings
    
    return df
