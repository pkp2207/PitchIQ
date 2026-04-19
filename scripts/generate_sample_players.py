"""Generate a synthetic Transfermarkt-style players dataset for development.

Since the real Kaggle dataset (davidcariboo/player-scores) requires API
credentials and may not be available, this script creates a realistic
synthetic dataset of ~2000 players across the 323 national teams found in
the cleaned match data.

The generated data is saved to data/raw/players.csv with columns that
mirror the real Transfermarkt schema:
    player_id, name, country_of_citizenship, market_value_in_eur,
    position, date_of_birth, club_name

Usage:
    python scripts/generate_sample_players.py
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so we can read the features file for the team list
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSITIONS = [
    "Goalkeeper", "Centre-Back", "Left-Back", "Right-Back",
    "Defensive Midfield", "Central Midfield", "Attacking Midfield",
    "Left Winger", "Right Winger", "Centre-Forward",
]

# Distribution of positions in a typical squad of 23-26 players
POSITION_WEIGHTS = [3, 4, 2, 2, 2, 3, 2, 2, 2, 3]  # sums to ~25

# Tier-based market value ranges (EUR) — teams are assigned a tier
# that determines the distribution of player market values.
TIERS = {
    1: {"value_mean": 50_000_000, "value_std": 30_000_000, "label": "elite"},
    2: {"value_mean": 15_000_000, "value_std": 10_000_000, "label": "strong"},
    3: {"value_mean": 3_000_000,  "value_std": 2_000_000,  "label": "mid"},
    4: {"value_mean": 500_000,    "value_std": 400_000,     "label": "developing"},
    5: {"value_mean": 80_000,     "value_std": 60_000,      "label": "amateur"},
}

# Well-known elite teams (tier 1)
TIER1_TEAMS = {
    "Brazil", "Germany", "Argentina", "France", "Spain", "England",
    "Italy", "Netherlands", "Portugal", "Belgium",
}

# Strong teams (tier 2)
TIER2_TEAMS = {
    "Uruguay", "Colombia", "Croatia", "Denmark", "Switzerland", "Mexico",
    "United States", "Chile", "Sweden", "Japan", "South Korea", "Australia",
    "Nigeria", "Senegal", "Morocco", "Egypt", "Turkey", "Austria",
    "Czech Republic", "Poland", "Norway", "Scotland", "Wales", "Serbia",
    "Ukraine", "Russia", "Romania", "Hungary", "Peru", "Ecuador",
    "Paraguay", "Cameroon", "Ghana", "Ivory Coast", "Algeria", "Tunisia",
    "Iran", "Saudi Arabia", "Qatar", "Canada", "Republic of Ireland",
    "Greece", "Slovakia", "Slovenia", "Bosnia and Herzegovina",
    "North Macedonia", "Iceland", "Finland", "Costa Rica", "Panama",
    "Jamaica", "Israel",
}

# Mid-tier teams (tier 3) — any FIFA-recognized team not in 1, 2, or 5
TIER5_PATTERNS = {
    # Non-FIFA / micro / regional / unrecognized entities
    "Abkhazia", "Alderney", "Ambazonia", "Andalusia", "Arameans Suryoye",
    "Artsakh", "Aymara", "Barawa", "Basque Country", "Biafra", "Brittany",
    "Canary Islands", "Cascadia", "Catalonia", "Chameria", "Chechnya",
    "Cilento", "Corsica", "County of Nice", "Crimea", "Darfur",
    "Donetsk PR", "Délvidék", "Elba Island", "Ellan Vannin",
    "Falkland Islands", "Felvidék", "Franconia", "Frøya", "Galicia",
    "German DR", "Gotland", "Gozo", "Greenland", "Guernsey", "Găgăuzia",
    "Hitra", "Hmong", "Iraqi Kurdistan", "Isle of Man", "Isle of Wight",
    "Jersey", "Kabylia", "Kernow", "Kárpátalja", "Luhansk PR", "Madrid",
    "Mapuche", "Matabeleland", "Maule Sur", "Menorca", "Monaco",
    "Northern Cyprus", "Occitania", "Orkney", "Padania", "Panjab",
    "Parishes of Jersey", "Provence", "Raetia", "Republic of St. Pauli",
    "Rhodes", "Romani people", "Ryūkyū", "Réunion", "Saare County",
    "Saint Barthélemy", "Saint Helena", "Saint Martin",
    "Saint Pierre and Miquelon", "Sark", "Saugeais", "Sealand",
    "Seborga", "Shetland", "Silesia", "Somaliland", "South Ossetia",
    "Surrey", "Székely Land", "Sápmi", "Tamil Eelam", "Tibet", "Ticino",
    "Two Sicilies", "United Koreans in Japan", "Vatican City",
    "Wallis Islands and Futuna", "West Papua", "Western Armenia",
    "Western Isles", "Western Sahara", "Ynys Môn", "Yorkshire",
    "Yoruba Nation", "Yugoslavia", "Zanzibar", "Åland Islands",
    "Chagos Islands", "Czechoslovakia",
}

# Club names for flavor (used randomly)
CLUBS = [
    "FC Barcelona", "Real Madrid", "Manchester City", "Liverpool FC",
    "Bayern Munich", "Paris Saint-Germain", "Juventus", "Inter Milan",
    "Chelsea FC", "Arsenal FC", "Borussia Dortmund", "Atletico Madrid",
    "AC Milan", "Tottenham Hotspur", "Ajax Amsterdam", "Benfica",
    "Porto FC", "Sporting CP", "Napoli", "Lazio",
    "Roma", "Lyon", "Marseille", "Monaco FC",
    "Bayer Leverkusen", "RB Leipzig", "Villarreal", "Sevilla FC",
    "Leicester City", "Aston Villa", "Everton FC", "Wolverhampton",
    "Club América", "Boca Juniors", "River Plate", "Flamengo",
    "Palmeiras", "São Paulo FC", "Santos FC", "Grêmio",
    "Al Hilal", "Al Nassr", "Galatasaray", "Fenerbahce",
    "Zenit St. Petersburg", "Shakhtar Donetsk", "Celtic FC", "Rangers FC",
    "Anderlecht", "Club Brugge", "PSV Eindhoven", "Feyenoord",
    "Red Bull Salzburg", "Dinamo Zagreb", "Olympiacos", "PAOK",
    "Local Club FC", "Regional United", "City FC", "Town Athletic",
]

# First / last names pool for generating fake player names
FIRST_NAMES = [
    "Lucas", "Marco", "Ahmed", "Carlos", "James", "João", "Luis",
    "Mohamed", "Omar", "Chen", "Hiroshi", "Pierre", "Stefan", "Ivan",
    "Diego", "Roberto", "Andrei", "Yuki", "Kevin", "Thomas",
    "Fernando", "Gabriel", "Daniel", "Mateo", "Santiago", "Emre",
    "Kylian", "Ousmane", "Sadio", "Victor", "Alex", "Samuel",
    "David", "Manuel", "Oscar", "Rafael", "Paulo", "Andre",
    "Sergio", "Ricardo", "Nikolai", "Hassan", "Ibrahim", "Kwame",
    "Sekou", "Tariq", "Liam", "Noah", "Elias", "Aleksandar",
]

LAST_NAMES = [
    "Silva", "Müller", "Santos", "Kim", "Park", "Hernandez", "Martinez",
    "Lopez", "Garcia", "Rodriguez", "Nguyen", "Ahmed", "Ali", "Wang",
    "Li", "Petrov", "Ivanov", "Johansson", "Nielsen", "Andersen",
    "Fischer", "Schneider", "Weber", "Schmidt", "Bauer", "Costa",
    "Oliveira", "Ferreira", "Almeida", "Pereira", "Diallo", "Traoré",
    "Touré", "Mbaye", "Diop", "Mendy", "Keita", "Camara",
    "Tanaka", "Suzuki", "Takahashi", "Nakamura", "Yamamoto",
    "Okafor", "Eze", "Adeyemi", "Akinola", "Salah", "Hassan",
    "Johnson", "Williams", "Brown", "Jones", "Smith",
]


def assign_tier(team_name: str) -> int:
    """Assign a market-value tier (1-5) based on team prestige."""
    if team_name in TIER1_TEAMS:
        return 1
    if team_name in TIER2_TEAMS:
        return 2
    if team_name in TIER5_PATTERNS:
        return 5
    # Default to tier 3 (developing FIFA members) or tier 4
    # Use a simple heuristic: shorter known team names are more likely tier 3
    return random.choices([3, 4], weights=[0.6, 0.4])[0]


def generate_players(teams: list, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic player data for all given teams."""
    rng = np.random.RandomState(seed)
    random.seed(seed)

    records = []
    player_id = 1

    for team in teams:
        tier = assign_tier(team)
        tier_info = TIERS[tier]

        # Squad size: elite teams get 23-26, amateur/micro get 15-20
        if tier <= 2:
            squad_size = rng.randint(23, 27)
        elif tier == 3:
            squad_size = rng.randint(20, 26)
        elif tier == 4:
            squad_size = rng.randint(18, 24)
        else:
            squad_size = rng.randint(15, 21)

        # Generate positions with realistic distribution
        positions = []
        for pos, weight in zip(POSITIONS, POSITION_WEIGHTS):
            count = max(1, round(weight * squad_size / sum(POSITION_WEIGHTS)))
            positions.extend([pos] * count)
        # Trim or pad to exact squad_size
        if len(positions) > squad_size:
            positions = rng.choice(positions, size=squad_size, replace=False).tolist()
        while len(positions) < squad_size:
            positions.append(rng.choice(POSITIONS))

        for pos in positions:
            # Market value: log-normal-ish distribution with tier-based parameters
            raw_value = max(10_000, rng.normal(tier_info["value_mean"], tier_info["value_std"]))
            # Position multiplier: attackers and midfielders tend to be valued higher
            if pos in ("Centre-Forward", "Attacking Midfield", "Left Winger", "Right Winger"):
                raw_value *= rng.uniform(1.0, 1.4)
            elif pos == "Goalkeeper":
                raw_value *= rng.uniform(0.5, 0.8)
            market_value = int(max(10_000, raw_value))

            # Age: most players 18-35, with peak around 25-28
            age = int(np.clip(rng.normal(26, 4), 17, 38))
            # date_of_birth — approximate, use a reference year of 2023
            birth_year = 2023 - age
            birth_month = rng.randint(1, 13)
            birth_day = rng.randint(1, 29)  # safe for all months
            dob = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"

            # Name
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            name = f"{first} {last}"

            # Club — elite players go to top clubs, lower tiers to regional clubs
            if tier <= 2:
                club = random.choice(CLUBS[:30])
            elif tier == 3:
                club = random.choice(CLUBS[:45])
            else:
                club = random.choice(CLUBS[40:])

            records.append({
                "player_id": player_id,
                "name": name,
                "country_of_citizenship": team,
                "market_value_in_eur": market_value,
                "position": pos,
                "date_of_birth": dob,
                "club_name": club,
            })
            player_id += 1

    return pd.DataFrame(records)


def main():
    # Read teams from the processed features file
    features_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if features_path.exists():
        df = pd.read_csv(features_path)
        teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    else:
        # Fallback: read from raw results
        raw_path = PROJECT_ROOT / "dataset" / "results.csv"
        if not raw_path.exists():
            raw_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
        df = pd.read_csv(raw_path)
        teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))

    print(f"Generating synthetic player data for {len(teams)} teams...")
    players_df = generate_players(teams)

    out_dir = PROJECT_ROOT / "data" / "raw"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "players.csv"
    players_df.to_csv(out_path, index=False)
    print(f"Saved {len(players_df)} players to {out_path}")
    print(f"Sample:\n{players_df.head(10).to_string()}")


if __name__ == "__main__":
    main()
