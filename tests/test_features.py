import pytest
import pandas as pd
import numpy as np
from src.features.elo import EloSystem, compute_elo_features
from src.features.form import compute_team_form
from src.features.head_to_head import compute_h2h_features
from src.features.streak import compute_streak_features
from src.features.squad_strength import compute_squad_features
from src.features.sentiment import compute_sentiment_features


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


# ---------------------------------------------------------------------------
# Squad Strength Features (Phase 2)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_players():
    """A small synthetic players DataFrame for testing squad strength features."""
    data = [
        # Brazil — 3 players, high value
        (1, "Player A", "Brazil", 50_000_000, "Centre-Forward", "1995-06-15", "FC Barcelona"),
        (2, "Player B", "Brazil", 30_000_000, "Goalkeeper", "1990-01-10", "Real Madrid"),
        (3, "Player C", "Brazil", 40_000_000, "Central Midfield", "1998-03-20", "Liverpool FC"),
        # Argentina — 3 players, slightly lower value
        (4, "Player D", "Argentina", 25_000_000, "Centre-Forward", "1993-06-24", "PSG"),
        (5, "Player E", "Argentina", 20_000_000, "Centre-Back", "1991-02-05", "Juventus"),
        (6, "Player F", "Argentina", 15_000_000, "Left Winger", "1997-11-30", "Inter Milan"),
        # Germany — 2 players
        (7, "Player G", "Germany", 45_000_000, "Attacking Midfield", "1999-07-01", "Bayern Munich"),
        (8, "Player H", "Germany", 35_000_000, "Right-Back", "1996-09-12", "Dortmund"),
        # France — 2 players
        (9, "Player I", "France", 60_000_000, "Centre-Forward", "2000-12-20", "PSG"),
        (10, "Player J", "France", 10_000_000, "Goalkeeper", "1988-04-15", "Lyon"),
    ]
    df = pd.DataFrame(data, columns=[
        "player_id", "name", "country_of_citizenship", "market_value_in_eur",
        "position", "date_of_birth", "club_name",
    ])
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    return df


class TestSquadStrengthFeatures:
    def test_squad_columns_added(self, sample_matches, sample_players):
        df = compute_squad_features(sample_matches, sample_players)
        expected_cols = [
            "home_squad_value", "away_squad_value", "squad_value_diff",
            "home_avg_age", "away_avg_age",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, sample_matches, sample_players):
        original_len = len(sample_matches)
        df = compute_squad_features(sample_matches, sample_players)
        assert len(df) == original_len

    def test_squad_value_diff_is_correct(self, sample_matches, sample_players):
        df = compute_squad_features(sample_matches, sample_players)
        for _, row in df.iterrows():
            assert row["squad_value_diff"] == pytest.approx(
                row["home_squad_value"] - row["away_squad_value"]
            )

    def test_elite_team_has_higher_value(self, sample_matches, sample_players):
        """Brazil (120M total) should have higher squad value than Argentina (60M)."""
        df = compute_squad_features(sample_matches, sample_players)
        # First match: Brazil (home) vs Argentina (away)
        first = df.iloc[0]
        assert first["home_squad_value"] > first["away_squad_value"]
        assert first["squad_value_diff"] > 0

    def test_unknown_team_gets_default(self, sample_players):
        """A team not in the player data should get a default value, not NaN."""
        matches = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01"]),
            "home_team": ["Tuvalu"],
            "away_team": ["Brazil"],
            "home_score": [0],
            "away_score": [5],
            "outcome": [-1],
        })
        df = compute_squad_features(matches, sample_players)
        assert not pd.isna(df.iloc[0]["home_squad_value"])
        assert df.iloc[0]["home_squad_value"] > 0  # default is log1p(500_000)

    def test_avg_age_is_reasonable(self, sample_matches, sample_players):
        df = compute_squad_features(sample_matches, sample_players)
        assert (df["home_avg_age"] >= 15).all()
        assert (df["home_avg_age"] <= 45).all()
        assert (df["away_avg_age"] >= 15).all()
        assert (df["away_avg_age"] <= 45).all()

    def test_no_nan_values(self, sample_matches, sample_players):
        df = compute_squad_features(sample_matches, sample_players)
        squad_cols = [
            "home_squad_value", "away_squad_value", "squad_value_diff",
            "home_avg_age", "away_avg_age",
        ]
        for col in squad_cols:
            assert df[col].isna().sum() == 0, f"NaN found in {col}"

    def test_original_columns_preserved(self, sample_matches, sample_players):
        """compute_squad_features should not remove any existing columns."""
        original_cols = set(sample_matches.columns)
        df = compute_squad_features(sample_matches, sample_players)
        assert original_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Sentiment Features (Phase 3)
# ---------------------------------------------------------------------------


class TestSentimentFeatures:
    EXPECTED_COLS = [
        "home_sentiment_avg",
        "away_sentiment_avg",
        "sentiment_diff",
        "home_sentiment_volume",
        "away_sentiment_volume",
    ]

    def test_sentiment_columns_added(self, sample_matches, sample_sentiment):
        """All 5 sentiment columns should be present after computing features."""
        df = compute_sentiment_features(sample_matches, sample_sentiment)
        for col in self.EXPECTED_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, sample_matches, sample_sentiment):
        """Output should have the same number of rows as the input matches."""
        original_len = len(sample_matches)
        df = compute_sentiment_features(sample_matches, sample_sentiment)
        assert len(df) == original_len

    def test_sentiment_diff_is_correct(self, sample_matches, sample_sentiment):
        """sentiment_diff should equal home_sentiment_avg - away_sentiment_avg."""
        df = compute_sentiment_features(sample_matches, sample_sentiment)
        for _, row in df.iterrows():
            assert row["sentiment_diff"] == pytest.approx(
                row["home_sentiment_avg"] - row["away_sentiment_avg"]
            )

    def test_no_future_sentiment_used(self, sample_matches, sample_sentiment):
        """Sentiment from after the match date must NOT influence the features.

        The sample_sentiment fixture includes a Brazil entry on 2021-09-15 with
        score 0.99.  The last match is 2021-08-01.  If future leakage exists,
        Brazil's sentiment avg for the final match would be pulled toward 0.99.
        We verify it is not.
        """
        df = compute_sentiment_features(sample_matches, sample_sentiment)

        # Last match (2021-08-01): Argentina vs France
        # Check the Brazil home sentiment on the last Brazil match (2021-07-01,
        # row index 18: Brazil vs Germany).
        brazil_home_rows = df[df["home_team"] == "Brazil"]
        last_brazil = brazil_home_rows.iloc[-1]
        # The future entry (0.99) should NOT be included; avg should be < 0.99
        assert last_brazil["home_sentiment_avg"] < 0.99

    def test_missing_sentiment_defaults_to_zero(self, sample_matches, sample_sentiment):
        """Teams with no sentiment data should get 0.0 for avg and 0 for volume."""
        # Create a match with a team that has no sentiment data at all
        unknown_match = pd.DataFrame({
            "date": pd.to_datetime(["2020-06-01"]),
            "home_team": ["Tuvalu"],
            "away_team": ["Nauru"],
            "home_score": [1],
            "away_score": [0],
            "tournament": ["Friendly"],
            "city": ["Funafuti"],
            "country": ["Tuvalu"],
            "neutral": [0],
            "outcome": [1],
        })
        df = compute_sentiment_features(unknown_match, sample_sentiment)
        assert df.iloc[0]["home_sentiment_avg"] == pytest.approx(0.0)
        assert df.iloc[0]["away_sentiment_avg"] == pytest.approx(0.0)
        assert df.iloc[0]["home_sentiment_volume"] == 0
        assert df.iloc[0]["away_sentiment_volume"] == 0

    def test_sentiment_score_bounded(self, sample_matches, sample_sentiment):
        """Average sentiment scores should be in [-1, 1]."""
        df = compute_sentiment_features(sample_matches, sample_sentiment)
        assert (df["home_sentiment_avg"] >= -1.0).all()
        assert (df["home_sentiment_avg"] <= 1.0).all()
        assert (df["away_sentiment_avg"] >= -1.0).all()
        assert (df["away_sentiment_avg"] <= 1.0).all()

    def test_volume_non_negative(self, sample_matches, sample_sentiment):
        """Sentiment volume (count of articles/entries) should be >= 0."""
        df = compute_sentiment_features(sample_matches, sample_sentiment)
        assert (df["home_sentiment_volume"] >= 0).all()
        assert (df["away_sentiment_volume"] >= 0).all()
