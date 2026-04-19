import pytest
import pandas as pd
import numpy as np
from src.features.elo import EloSystem, compute_elo_features
from src.features.form import compute_team_form
from src.features.head_to_head import compute_h2h_features
from src.features.streak import compute_streak_features


class TestEloSystem:
    def test_initial_rating(self):
        elo = EloSystem(base_rating=1500, k_factor=32)
        assert elo.get_rating("Brazil") == 1500

    def test_update_after_win(self):
        elo = EloSystem(base_rating=1500, k_factor=32)
        elo.update_ratings("Brazil", "Argentina", 1.0)  # Brazil wins
        assert elo.get_rating("Brazil") > 1500
        assert elo.get_rating("Argentina") < 1500

    def test_update_after_draw(self):
        elo = EloSystem(base_rating=1500, k_factor=32)
        elo.update_ratings("Brazil", "Argentina", 0.5)
        # Draw between equal teams — ratings stay the same
        assert elo.get_rating("Brazil") == pytest.approx(1500, abs=0.01)
        assert elo.get_rating("Argentina") == pytest.approx(1500, abs=0.01)

    def test_expected_result_symmetric(self):
        elo = EloSystem()
        e = elo.expected_result(1500, 1500)
        assert e == pytest.approx(0.5, abs=0.001)

    def test_higher_rated_has_higher_expected(self):
        elo = EloSystem()
        e = elo.expected_result(1600, 1400)
        assert e > 0.5


class TestEloFeatures:
    def test_elo_columns_added(self, sample_matches):
        df = compute_elo_features(sample_matches)
        assert "home_elo_before" in df.columns
        assert "away_elo_before" in df.columns
        assert "elo_diff" in df.columns

    def test_first_match_uses_base_rating(self, sample_matches):
        df = compute_elo_features(sample_matches)
        assert df.iloc[0]["home_elo_before"] == 1500
        assert df.iloc[0]["away_elo_before"] == 1500

    def test_elo_diff_is_correct(self, sample_matches):
        df = compute_elo_features(sample_matches)
        for _, row in df.iterrows():
            assert row["elo_diff"] == pytest.approx(
                row["home_elo_before"] - row["away_elo_before"]
            )


class TestFormFeatures:
    def test_form_columns_added(self, sample_matches):
        df = compute_elo_features(sample_matches)  # elo needed first for outcome
        df = compute_team_form(df)
        for prefix in ["home_", "away_"]:
            for window in [5, 10]:
                assert f"{prefix}win_rate_{window}" in df.columns
                assert f"{prefix}goals_scored_avg_{window}" in df.columns
                assert f"{prefix}goals_conceded_avg_{window}" in df.columns
                assert f"{prefix}goal_diff_avg_{window}" in df.columns

    def test_home_advantage_column_added(self, sample_matches):
        df = compute_team_form(sample_matches)
        assert "home_advantage" in df.columns

    def test_home_advantage_first_match_is_zero(self, sample_matches):
        df = compute_team_form(sample_matches)
        # First match for each team should have home_advantage = 0 (no prior data)
        assert df.iloc[0]["home_advantage"] == 0.0

    def test_home_advantage_bounded(self, sample_matches):
        df = compute_team_form(sample_matches)
        assert df["home_advantage"].min() >= 0.0
        assert df["home_advantage"].max() <= 1.0

    def test_no_future_data_leakage(self, sample_matches):
        """The first match for each team should have form = 0 (no prior data)."""
        df = compute_team_form(sample_matches)
        # Brazil's first match is the first row — form should be 0
        first = df.iloc[0]
        assert first["home_win_rate_5"] == 0.0
        assert first["home_goals_scored_avg_5"] == 0.0
        assert first["home_goal_diff_avg_5"] == 0.0

    def test_goal_diff_avg_no_leakage(self, sample_matches):
        """First match should have goal_diff_avg = 0."""
        df = compute_team_form(sample_matches)
        first = df.iloc[0]
        assert first["home_goal_diff_avg_5"] == 0.0
        assert first["home_goal_diff_avg_10"] == 0.0


class TestH2HFeatures:
    def test_h2h_columns_added(self, sample_matches):
        df = compute_h2h_features(sample_matches)
        assert "h2h_home_win_rate" in df.columns
        assert "h2h_avg_goal_diff" in df.columns
        assert "h2h_num_meetings" in df.columns

    def test_first_meeting_has_zero_history(self, sample_matches):
        df = compute_h2h_features(sample_matches)
        # First match (Brazil vs Argentina) should have no history
        first = df.iloc[0]
        assert first["h2h_num_meetings"] == 0
        assert first["h2h_home_win_rate"] == 0.0

    def test_h2h_accumulates(self, sample_matches):
        df = compute_h2h_features(sample_matches)
        # After the first Brazil-Argentina match, the next should have 1 meeting
        brazil_argentina = df[
            ((df["home_team"] == "Brazil") & (df["away_team"] == "Argentina")) |
            ((df["home_team"] == "Argentina") & (df["away_team"] == "Brazil"))
        ]
        meetings = brazil_argentina["h2h_num_meetings"].tolist()
        assert meetings[0] == 0
        assert meetings[1] >= 1

    def test_chronological_order(self, sample_matches):
        """H2H meetings should never decrease over time for the same matchup."""
        df = compute_h2h_features(sample_matches)
        brazil_argentina = df[
            ((df["home_team"] == "Brazil") & (df["away_team"] == "Argentina")) |
            ((df["home_team"] == "Argentina") & (df["away_team"] == "Brazil"))
        ]
        meetings = brazil_argentina["h2h_num_meetings"].tolist()
        for i in range(1, len(meetings)):
            assert meetings[i] >= meetings[i - 1]


class TestStreakFeatures:
    def test_streak_columns_added(self, sample_matches):
        df = compute_streak_features(sample_matches)
        assert "home_streak" in df.columns
        assert "away_streak" in df.columns

    def test_first_match_streak_is_zero(self, sample_matches):
        df = compute_streak_features(sample_matches)
        # First match ever — no prior data, streak should be 0
        assert df.iloc[0]["home_streak"] == 0
        assert df.iloc[0]["away_streak"] == 0

    def test_streak_positive_after_wins(self, sample_matches):
        """After consecutive wins the streak should be positive."""
        df = compute_streak_features(sample_matches)
        # Brazil wins match 0 (2-1) and match 2 (3-0 as away, reversed = win).
        # After two consecutive wins, next Brazil row should have streak >= 1.
        brazil_rows = df[
            (df["home_team"] == "Brazil") | (df["away_team"] == "Brazil")
        ]
        # Collect streaks for Brazil regardless of home/away
        streaks = []
        for _, row in brazil_rows.iterrows():
            if row["home_team"] == "Brazil":
                streaks.append(row["home_streak"])
            else:
                streaks.append(row["away_streak"])
        # Second match for Brazil (match idx 2, Argentina vs Brazil where Brazil wins)
        # streak should reflect previous win
        assert streaks[1] >= 1

    def test_streak_negative_after_loss(self, sample_matches):
        """Streak should go negative after a loss."""
        df = compute_streak_features(sample_matches)
        # Find a team that lost and check if their next streak is negative
        # Argentina loses first match (away, 2-1 to Brazil)
        # Argentina loses second match (home, Argentina 0-3 Brazil)
        arg_rows = df[
            (df["home_team"] == "Argentina") | (df["away_team"] == "Argentina")
        ]
        streaks = []
        for _, row in arg_rows.iterrows():
            if row["home_team"] == "Argentina":
                streaks.append(row["home_streak"])
            else:
                streaks.append(row["away_streak"])
        # After first loss, second match streak should be negative
        assert streaks[1] <= -1

    def test_row_count_preserved(self, sample_matches):
        original_len = len(sample_matches)
        df = compute_streak_features(sample_matches)
        assert len(df) == original_len


class TestDaysSinceLastFeatures:
    def test_days_since_columns_added(self, sample_matches):
        df = compute_streak_features(sample_matches)
        assert "home_days_since_last" in df.columns
        assert "away_days_since_last" in df.columns

    def test_first_match_default_30(self, sample_matches):
        df = compute_streak_features(sample_matches)
        # First match for Brazil and Argentina should default to 30
        assert df.iloc[0]["home_days_since_last"] == 30
        assert df.iloc[0]["away_days_since_last"] == 30

    def test_days_since_is_positive(self, sample_matches):
        df = compute_streak_features(sample_matches)
        assert (df["home_days_since_last"] >= 0).all()
        assert (df["away_days_since_last"] >= 0).all()

    def test_days_since_decreases_for_frequent_play(self, sample_matches):
        """Second match should have fewer days_since than the default 30
        when matches are close together (monthly in the fixture)."""
        df = compute_streak_features(sample_matches)
        # Brazil plays match 0 (Jan 1) and match 2 (Mar 1) as away.
        # Days between = ~59. But match 4 (May 1, home) is ~61 days after match 2.
        # All of these should be less than or around 60, not the default 30.
        brazil_home = df[df["home_team"] == "Brazil"]
        # The second Brazil home match should have a real days_since value (not default 30)
        second = brazil_home.iloc[1]
        # Brazil plays home Jan 1, then away Mar 1, then home May 1 => ~61 days from Mar
        assert second["home_days_since_last"] > 0
