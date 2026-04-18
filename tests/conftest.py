import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_matches():
    """A small DataFrame of ~20 matches with known outcomes for testing."""
    data = [
        ("2020-01-01", "Brazil", "Argentina", 2, 1, "Friendly", "Rio de Janeiro", "Brazil", False),
        ("2020-02-01", "Germany", "France", 1, 1, "Friendly", "Berlin", "Germany", False),
        ("2020-03-01", "Argentina", "Brazil", 0, 3, "Copa America", "Buenos Aires", "Argentina", False),
        ("2020-04-01", "France", "Germany", 2, 0, "UEFA Euro", "Paris", "France", False),
        ("2020-05-01", "Brazil", "Germany", 1, 0, "Friendly", "Rio de Janeiro", "Brazil", False),
        ("2020-06-01", "Argentina", "France", 1, 2, "Friendly", "Buenos Aires", "Argentina", False),
        ("2020-07-01", "Germany", "Brazil", 3, 1, "Friendly", "Berlin", "Germany", False),
        ("2020-08-01", "France", "Argentina", 1, 0, "Friendly", "Paris", "France", False),
        ("2020-09-01", "Brazil", "France", 0, 0, "Friendly", "Zurich", "Zurich", True),
        ("2020-10-01", "Germany", "Argentina", 2, 2, "Friendly", "Berlin", "Germany", False),
        ("2020-11-01", "Brazil", "Argentina", 1, 0, "FIFA World Cup qualification", "Rio de Janeiro", "Brazil", False),
        ("2020-12-01", "France", "Germany", 3, 1, "UEFA Euro qualification", "Paris", "France", False),
        ("2021-01-01", "Argentina", "Germany", 2, 1, "Friendly", "Buenos Aires", "Argentina", False),
        ("2021-02-01", "Brazil", "France", 2, 2, "Friendly", "Rio de Janeiro", "Brazil", False),
        ("2021-03-01", "Germany", "France", 0, 1, "Friendly", "Berlin", "Germany", False),
        ("2021-04-01", "Argentina", "Brazil", 1, 1, "Copa America", "Buenos Aires", "Argentina", False),
        ("2021-05-01", "France", "Brazil", 2, 0, "Friendly", "Paris", "France", False),
        ("2021-06-01", "Germany", "Argentina", 1, 3, "Friendly", "Berlin", "Germany", False),
        ("2021-07-01", "Brazil", "Germany", 4, 0, "Friendly", "Rio de Janeiro", "Brazil", False),
        ("2021-08-01", "Argentina", "France", 0, 1, "Friendly", "Buenos Aires", "Argentina", False),
    ]

    df = pd.DataFrame(data, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(int)

    # Encode outcome (home perspective)
    def outcome(row):
        if row["home_score"] > row["away_score"]:
            return 1
        elif row["home_score"] == row["away_score"]:
            return 0
        return -1

    df["outcome"] = df.apply(outcome, axis=1)
    return df
