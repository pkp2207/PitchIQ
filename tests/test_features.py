import pytest
import pandas as pd
import numpy as np
from src.features.elo import EloSystem, compute_elo_features
from src.features.form import compute_team_form
from src.features.head_to_head import compute_h2h_features


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

    def test_no_future_data_leakage(self, sample_matches):
        """The first match for each team should have form = 0 (no prior data)."""
        df = compute_team_form(sample_matches)
        # Brazil's first match is the first row — form should be 0
        first = df.iloc[0]
        assert first["home_win_rate_5"] == 0.0
        assert first["home_goals_scored_avg_5"] == 0.0


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
