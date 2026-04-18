"""
Generates a synthetic results.csv (~500 rows) for development and testing.
Uses real team names, realistic score distributions, and date range 1990-2024.
"""
import os
import random
import csv
from pathlib import Path
from datetime import date, timedelta

TEAMS = [
    "Brazil", "Argentina", "France", "Germany", "England",
    "Spain", "Italy", "Netherlands", "Portugal", "Belgium",
    "Uruguay", "Colombia", "Mexico", "Japan", "South Korea",
    "United States", "Australia", "Nigeria", "Cameroon", "Egypt",
]

TOURNAMENTS = [
    "Friendly", "Friendly", "Friendly",  # weighted toward friendlies
    "FIFA World Cup", "FIFA World Cup qualification",
    "Copa America", "UEFA Euro", "UEFA Euro qualification",
    "AFC Asian Cup", "Africa Cup of Nations",
]

CITIES = {
    "Brazil": "Rio de Janeiro", "Argentina": "Buenos Aires", "France": "Paris",
    "Germany": "Berlin", "England": "London", "Spain": "Madrid",
    "Italy": "Rome", "Netherlands": "Amsterdam", "Portugal": "Lisbon",
    "Belgium": "Brussels", "Uruguay": "Montevideo", "Colombia": "Bogota",
    "Mexico": "Mexico City", "Japan": "Tokyo", "South Korea": "Seoul",
    "United States": "New York", "Australia": "Sydney", "Nigeria": "Lagos",
    "Cameroon": "Yaounde", "Egypt": "Cairo",
}

NEUTRAL_CITIES = ["Zurich", "Doha", "Singapore", "Dubai", "Johannesburg"]


def generate_sample_data(n_matches: int = 500, seed: int = 42) -> list[dict]:
    random.seed(seed)
    matches = []
    start_date = date(1990, 1, 1)
    end_date = date(2024, 12, 31)
    total_days = (end_date - start_date).days

    for _ in range(n_matches):
        home, away = random.sample(TEAMS, 2)
        tournament = random.choice(TOURNAMENTS)
        match_date = start_date + timedelta(days=random.randint(0, total_days))

        # ~20% neutral venue
        neutral = random.random() < 0.2
        if neutral:
            city = random.choice(NEUTRAL_CITIES)
            country = city  # simplified
        else:
            city = CITIES.get(home, "Unknown")
            country = home

        # Realistic score distribution: most goals 0-3, occasional higher
        home_score = random.choices(range(6), weights=[20, 30, 25, 15, 7, 3])[0]
        away_score = random.choices(range(6), weights=[25, 30, 22, 14, 6, 3])[0]

        matches.append({
            "date": match_date.isoformat(),
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": tournament,
            "city": city,
            "country": country,
            "neutral": str(neutral).upper(),
        })

    # Sort chronologically
    matches.sort(key=lambda m: m["date"])
    return matches


def main():
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "raw"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "results.csv"

    matches = generate_sample_data(500)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "neutral",
        ])
        writer.writeheader()
        writer.writerows(matches)

    print(f"Generated {len(matches)} matches -> {out_path}")


if __name__ == "__main__":
    main()
