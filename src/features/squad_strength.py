"""Phase 2 — Transfermarkt squad strength features.

Computes per-match squad-level features by looking up player data for both
teams' countries.  For each match this produces:

    home_squad_value      — log-scaled total squad market value (home team)
    away_squad_value      — log-scaled total squad market value (away team)
    squad_value_diff      — home_squad_value minus away_squad_value
    home_avg_age          — mean age of the home team's squad (in years)
    away_avg_age          — mean age of the away team's squad (in years)

Time-awareness note
-------------------
The player data is treated as a **static snapshot** — the same squad stats
are applied to every match for a given national team regardless of when the
match took place.  In a real production system you would want player data
that is versioned by season/year so that market values and ages reflect the
state at match time.  This is noted here and is acceptable for the
Phase 2 synthetic-data implementation.
"""

import numpy as np
import pandas as pd


def _build_squad_lookup(players_df: pd.DataFrame, reference_date: pd.Timestamp = None) -> dict:
    """Pre-compute per-country squad aggregates from the player data.

    Parameters
    ----------
    players_df : DataFrame
        Must contain at least ``country_of_citizenship`` and
        ``market_value_in_eur``.  Optionally ``date_of_birth`` for age.
    reference_date : Timestamp, optional
        Date used to compute player age.  Defaults to 2023-06-01 (mid-year)
        which is a reasonable reference for the synthetic snapshot.

    Returns
    -------
    dict
        Mapping from country name to a dict with keys:
        ``total_value``, ``log_total_value``, ``avg_age``, ``squad_size``.
    """
    if reference_date is None:
        reference_date = pd.Timestamp("2023-06-01")

    lookup = {}

    for country, group in players_df.groupby("country_of_citizenship"):
        total_value = group["market_value_in_eur"].sum()

        # Log-scale total value to reduce skew (elite teams have 100x+
        # the value of amateur teams).  Adding 1 avoids log(0).
        log_total_value = np.log1p(total_value)

        # Average age — compute from date_of_birth if available
        if "date_of_birth" in group.columns:
            dob = pd.to_datetime(group["date_of_birth"], errors="coerce")
            ages = (reference_date - dob).dt.days / 365.25
            avg_age = ages.mean()
            if pd.isna(avg_age):
                avg_age = 26.0  # sensible default
        else:
            avg_age = 26.0

        lookup[country] = {
            "total_value": total_value,
            "log_total_value": log_total_value,
            "avg_age": round(avg_age, 2),
            "squad_size": len(group),
        }

    return lookup


def compute_squad_features(matches_df: pd.DataFrame,
                           players_df: pd.DataFrame) -> pd.DataFrame:
    """Add squad-strength features to the match DataFrame.

    Parameters
    ----------
    matches_df : DataFrame
        Must contain ``home_team`` and ``away_team`` columns.
    players_df : DataFrame
        Player-level data with ``country_of_citizenship``,
        ``market_value_in_eur``, and optionally ``date_of_birth``.

    Returns
    -------
    DataFrame
        The input ``matches_df`` with five new columns appended:
        ``home_squad_value``, ``away_squad_value``, ``squad_value_diff``,
        ``home_avg_age``, ``away_avg_age``.
    """
    # Build lookup once — O(players) instead of O(matches * players)
    lookup = _build_squad_lookup(players_df)

    # Default values for teams not found in the player data
    default_log_value = np.log1p(500_000)  # modest default
    default_avg_age = 26.0

    home_values = []
    away_values = []
    home_ages = []
    away_ages = []

    for _, row in matches_df.iterrows():
        h = lookup.get(row["home_team"])
        a = lookup.get(row["away_team"])

        home_values.append(h["log_total_value"] if h else default_log_value)
        away_values.append(a["log_total_value"] if a else default_log_value)
        home_ages.append(h["avg_age"] if h else default_avg_age)
        away_ages.append(a["avg_age"] if a else default_avg_age)

    matches_df = matches_df.copy()
    matches_df["home_squad_value"] = home_values
    matches_df["away_squad_value"] = away_values
    matches_df["squad_value_diff"] = (
        matches_df["home_squad_value"] - matches_df["away_squad_value"]
    )
    matches_df["home_avg_age"] = home_ages
    matches_df["away_avg_age"] = away_ages

    return matches_df
