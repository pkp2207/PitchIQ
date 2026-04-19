"""Generate a synthetic sentiment dataset for Phase 3 development.

Since real news/social-media data is not available, this script creates a
realistic synthetic dataset of ~10,000 sentiment entries tied to matches in
the processed features file.

For each match, 1-3 sentiment entries are generated per team in the 7 days
before the match date.  Teams with higher Elo tend toward slightly more
positive sentiment, but random variation ensures underdogs sometimes get
positive buzz and favourites sometimes get negative press.

The generated data is saved to data/raw/sentiment.csv with columns:
    date, team, sentiment_score, source, headline

Usage:
    python scripts/generate_sample_sentiment.py
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCES = ["news", "social_media", "expert_opinion"]

# Headline templates — {team} will be replaced at generation time
POSITIVE_HEADLINES = [
    "{team} looking strong ahead of {tournament}",
    "{team} squad in high spirits before big match",
    "Fans optimistic as {team} prepare for clash",
    "{team} star player declared fit for upcoming game",
    "Analysts predict strong showing from {team}",
    "{team} morale high after solid training sessions",
    "Media praises {team} tactical preparation",
    "{team} coach confident in squad depth",
    "Experts tip {team} as favourites for the fixture",
    "{team} riding wave of positive momentum",
]

NEGATIVE_HEADLINES = [
    "{team} hit by injury crisis before crucial match",
    "Concern grows as {team} struggle in training",
    "{team} fans frustrated with recent poor form",
    "Key {team} player doubtful for upcoming game",
    "Analysts worried about {team} defensive frailty",
    "Internal tensions reported in {team} camp",
    "Media questions {team} tactical approach",
    "{team} coach under pressure after disappointing results",
    "Experts predict tough outing for {team}",
    "{team} confidence dented by off-pitch issues",
]

NEUTRAL_HEADLINES = [
    "{team} finalise preparations for {tournament} match",
    "{team} hold closed-door training session",
    "{team} announce squad for upcoming fixture",
    "Steady build-up continues for {team} ahead of game",
    "{team} players remain focused on the task ahead",
]


def _pick_headline(sentiment_score: float, team: str, tournament: str) -> str:
    """Select a headline template based on sentiment polarity."""
    if sentiment_score > 0.2:
        template = random.choice(POSITIVE_HEADLINES)
    elif sentiment_score < -0.2:
        template = random.choice(NEGATIVE_HEADLINES)
    else:
        template = random.choice(NEUTRAL_HEADLINES)
    return template.format(team=team, tournament=tournament)


def generate_sentiment(features_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sentiment data for all matches in features_df.

    Parameters
    ----------
    features_df : DataFrame
        Must contain columns: date, home_team, away_team, tournament,
        home_elo_before, away_elo_before.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Columns: date, team, sentiment_score, source, headline
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    records = []

    # Normalise Elo values to [0, 1] range for sentiment bias
    elo_min = min(features_df["home_elo_before"].min(),
                  features_df["away_elo_before"].min())
    elo_max = max(features_df["home_elo_before"].max(),
                  features_df["away_elo_before"].max())
    elo_range = max(elo_max - elo_min, 1)  # avoid division by zero

    for _, row in features_df.iterrows():
        match_date = pd.to_datetime(row["date"])
        tournament = row["tournament"]

        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            elo = row[f"{side}_elo_before"]

            # Number of sentiment entries: 1-3 per team per match
            n_entries = rng.randint(1, 4)

            # Elo-based sentiment bias: higher Elo -> slightly positive
            elo_normalised = (elo - elo_min) / elo_range  # [0, 1]
            # Map to a small bias in [-0.15, 0.35] — higher-rated teams
            # tend toward positive, but the range is narrow enough that
            # noise dominates.
            elo_bias = (elo_normalised - 0.3) * 0.5  # approx [-0.15, 0.35]

            for _ in range(n_entries):
                # Date: 1-7 days before the match
                days_before = rng.randint(1, 8)
                entry_date = match_date - pd.Timedelta(days=int(days_before))

                # Sentiment score: elo_bias + noise, clipped to [-1, 1]
                noise = rng.normal(0, 0.35)
                sentiment_score = float(np.clip(elo_bias + noise, -1.0, 1.0))

                source = random.choice(SOURCES)
                headline = _pick_headline(sentiment_score, team, tournament)

                records.append({
                    "date": entry_date.strftime("%Y-%m-%d"),
                    "team": team,
                    "sentiment_score": round(sentiment_score, 4),
                    "source": source,
                    "headline": headline,
                })

    return pd.DataFrame(records)


def main():
    features_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not features_path.exists():
        # Fallback: try to run with raw data
        print(f"ERROR: {features_path} not found.")
        print("  Run the feature pipeline first: python -m src.features.pipeline")
        sys.exit(1)

    print("Loading features data...")
    features_df = pd.read_csv(features_path)
    features_df["date"] = pd.to_datetime(features_df["date"])

    # Sample a subset of matches to keep ~10,000 rows
    # With ~2 entries/team * 2 teams/match = ~4 entries/match, we need ~2500 matches
    n_matches = len(features_df)
    if n_matches > 2500:
        # Sample evenly across time
        sample_idx = np.linspace(0, n_matches - 1, 2500, dtype=int)
        sampled = features_df.iloc[sample_idx]
    else:
        sampled = features_df

    print(f"Generating sentiment data for {len(sampled)} matches...")
    sentiment_df = generate_sentiment(sampled, seed=42)

    out_dir = PROJECT_ROOT / "data" / "raw"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "sentiment.csv"
    sentiment_df.to_csv(out_path, index=False)

    print(f"Saved {len(sentiment_df)} sentiment entries to {out_path}")
    print(f"Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    print(f"Unique teams: {sentiment_df['team'].nunique()}")
    print(f"Source distribution:\n{sentiment_df['source'].value_counts().to_string()}")
    print(f"\nSample:\n{sentiment_df.head(10).to_string()}")


if __name__ == "__main__":
    main()
