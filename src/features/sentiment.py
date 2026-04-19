"""Phase 3 — Sentiment analysis features.

Computes per-match sentiment features by looking up pre-match sentiment
entries for both teams in the 7 days before each match.  Produces:

    home_sentiment_avg     — mean sentiment score for the home team
    away_sentiment_avg     — mean sentiment score for the away team
    sentiment_diff         — home_sentiment_avg minus away_sentiment_avg
    home_sentiment_volume  — number of sentiment entries (media attention proxy)
    away_sentiment_volume  — number of sentiment entries

Time-awareness note
-------------------
Sentiment entries are only considered if their date falls in the 7-day
window *before* the match date (exclusive of match day itself).  This
prevents data leakage from match-day or post-match reporting.

Efficiency note
---------------
The sentiment data is pre-grouped by team and sorted by date so that for
each match we can use binary search (via searchsorted) to find the
relevant window, avoiding O(matches * sentiment) brute force.
"""

import numpy as np
import pandas as pd


def _build_sentiment_lookup(sentiment_df: pd.DataFrame) -> dict:
    """Pre-group sentiment entries by team, sorted by date.

    Parameters
    ----------
    sentiment_df : DataFrame
        Must contain columns: date, team, sentiment_score.
        The ``date`` column should already be parsed as datetime.

    Returns
    -------
    dict
        Mapping from team name to a DataFrame (subset) sorted by date,
        with ``date`` as a numpy datetime64 array for fast searchsorted.
    """
    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    sentiment_df = sentiment_df.sort_values("date").reset_index(drop=True)

    lookup = {}
    for team, group in sentiment_df.groupby("team"):
        group = group.sort_values("date").reset_index(drop=True)
        lookup[team] = {
            "dates": group["date"].values,  # numpy datetime64 array
            "scores": group["sentiment_score"].values,
        }
    return lookup


def compute_sentiment_features(matches_df: pd.DataFrame,
                               sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment features to the match DataFrame.

    Parameters
    ----------
    matches_df : DataFrame
        Must contain ``date``, ``home_team``, ``away_team`` columns.
    sentiment_df : DataFrame
        Must contain ``date``, ``team``, ``sentiment_score`` columns.

    Returns
    -------
    DataFrame
        The input ``matches_df`` with five new columns appended:
        ``home_sentiment_avg``, ``away_sentiment_avg``, ``sentiment_diff``,
        ``home_sentiment_volume``, ``away_sentiment_volume``.
    """
    lookup = _build_sentiment_lookup(sentiment_df)

    matches_df = matches_df.copy()
    matches_df["date"] = pd.to_datetime(matches_df["date"])

    home_avgs = []
    away_avgs = []
    home_vols = []
    away_vols = []

    for _, row in matches_df.iterrows():
        match_date = row["date"]
        # 7-day window: (match_date - 7 days) to (match_date - 1 day) inclusive
        window_start = match_date - pd.Timedelta(days=7)
        window_end = match_date  # exclusive (sentiment must be before match day)

        for side, avgs_list, vols_list in [
            ("home_team", home_avgs, home_vols),
            ("away_team", away_avgs, away_vols),
        ]:
            team = row[side]
            team_data = lookup.get(team)

            if team_data is None:
                avgs_list.append(0.0)
                vols_list.append(0)
                continue

            dates = team_data["dates"]
            scores = team_data["scores"]

            # Binary search for window boundaries
            start_idx = np.searchsorted(dates, np.datetime64(window_start), side="left")
            end_idx = np.searchsorted(dates, np.datetime64(window_end), side="left")

            window_scores = scores[start_idx:end_idx]

            if len(window_scores) > 0:
                avgs_list.append(float(np.mean(window_scores)))
                vols_list.append(int(len(window_scores)))
            else:
                avgs_list.append(0.0)
                vols_list.append(0)

    matches_df["home_sentiment_avg"] = home_avgs
    matches_df["away_sentiment_avg"] = away_avgs
    matches_df["sentiment_diff"] = (
        matches_df["home_sentiment_avg"] - matches_df["away_sentiment_avg"]
    )
    matches_df["home_sentiment_volume"] = home_vols
    matches_df["away_sentiment_volume"] = away_vols

    return matches_df
