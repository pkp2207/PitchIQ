"""Utility functions for team name normalization and tournament weighting."""

import re

# Common aliases / misspellings -> canonical name
_TEAM_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "ir iran": "Iran",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "czech republic": "Czechia",
    "north macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
    "china pr": "China",
    "chinese taipei": "Taiwan",
}

# Tournament importance weights (higher = more important)
_TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.9,
    "Copa America": 0.9,
    "AFC Asian Cup": 0.85,
    "Africa Cup of Nations": 0.85,
    "FIFA World Cup qualification": 0.8,
    "UEFA Euro qualification": 0.75,
    "Confederations Cup": 0.7,
    "UEFA Nations League": 0.7,
    "Gold Cup": 0.7,
    "Friendly": 0.4,
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to a canonical form.

    Strips whitespace, fixes casing, and maps common aliases.
    """
    cleaned = name.strip()
    lookup = cleaned.lower()
    if lookup in _TEAM_ALIASES:
        return _TEAM_ALIASES[lookup]
    # Title-case as a reasonable default
    return cleaned.title() if cleaned == cleaned.lower() or cleaned == cleaned.upper() else cleaned


def tournament_to_weight(tournament: str) -> float:
    """Return a weight in [0, 1] indicating tournament importance.

    Returns 0.5 for unknown tournaments.
    """
    return _TOURNAMENT_WEIGHTS.get(tournament, 0.5)
