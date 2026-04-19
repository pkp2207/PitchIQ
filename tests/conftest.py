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


@pytest.fixture
def sample_sentiment():
    """A small sentiment DataFrame for testing sentiment features.

    Contains ~20 entries for Brazil, Argentina, France, and Germany with
    dates spanning 2019-12 through 2021-09 to cover the sample_matches
    fixture range (2020-01 to 2021-08).  Includes a mix of positive,
    negative, and neutral scores.
    """
    data = [
        # Brazil — mix of positive and neutral
        ("2019-12-28", "Brazil", 0.6, "headline_br_1"),
        ("2020-01-15", "Brazil", 0.3, "headline_br_2"),
        ("2020-04-20", "Brazil", -0.2, "headline_br_3"),
        ("2020-08-10", "Brazil", 0.8, "headline_br_4"),
        ("2021-02-15", "Brazil", 0.1, "headline_br_5"),
        # Argentina — mostly negative
        ("2020-01-10", "Argentina", -0.5, "headline_ar_1"),
        ("2020-03-15", "Argentina", -0.3, "headline_ar_2"),
        ("2020-06-20", "Argentina", 0.2, "headline_ar_3"),
        ("2020-10-05", "Argentina", -0.7, "headline_ar_4"),
        ("2021-05-20", "Argentina", -0.1, "headline_ar_5"),
        # France — positive sentiment
        ("2020-02-10", "France", 0.7, "headline_fr_1"),
        ("2020-05-15", "France", 0.4, "headline_fr_2"),
        ("2020-09-01", "France", 0.9, "headline_fr_3"),
        ("2021-01-10", "France", 0.5, "headline_fr_4"),
        ("2021-07-20", "France", 0.6, "headline_fr_5"),
        # Germany — mixed
        ("2020-01-20", "Germany", 0.0, "headline_de_1"),
        ("2020-04-10", "Germany", -0.4, "headline_de_2"),
        ("2020-07-05", "Germany", 0.5, "headline_de_3"),
        ("2020-11-15", "Germany", -0.6, "headline_de_4"),
        ("2021-06-10", "Germany", 0.3, "headline_de_5"),
        # Future sentiment (after all sample matches end 2021-08-01)
        ("2021-09-15", "Brazil", 0.99, "headline_br_future"),
    ]

    df = pd.DataFrame(data, columns=["date", "team", "sentiment_score", "headline"])
    df["date"] = pd.to_datetime(df["date"])
    return df
