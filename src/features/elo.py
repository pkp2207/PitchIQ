import pandas as pd

class EloSystem:
    def __init__(self, base_rating=1500, k_factor=32):
        self.ratings = {}
        self.base_rating = base_rating
        self.k_factor = k_factor
        
    def get_rating(self, team):
        return self.ratings.get(team, self.base_rating)
        
    def expected_result(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        
    def update_ratings(self, team_a, team_b, actual_result_a):
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        expected_a = self.expected_result(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        actual_result_b = 1 - actual_result_a
        
        new_rating_a = rating_a + self.k_factor * (actual_result_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_result_b - expected_b)
        
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b

def compute_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Elo ratings for all matches chronologically."""
    elo = EloSystem()
    
    home_elo_before = []
    away_elo_before = []
    
    # Sort chronologically just in case
    df = df.sort_values(by='date').reset_index(drop=True)
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        home_elo = elo.get_rating(home_team)
        away_elo = elo.get_rating(away_team)
        
        home_elo_before.append(home_elo)
        away_elo_before.append(away_elo)
        
        # Determine actual result for home team (1 for win, 0.5 for draw, 0 for loss)
        if row['outcome'] == 1:
            actual = 1
        elif row['outcome'] == 0:
            actual = 0.5
        else:
            actual = 0
            
        elo.update_ratings(home_team, away_team, actual)
        
    df['home_elo_before'] = home_elo_before
    df['away_elo_before'] = away_elo_before
    df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
    
    return df
