import streamlit as st
import pandas as pd
from datetime import datetime


def get_team_latest_stats(df: pd.DataFrame, team_name: str) -> dict:
    """Return the latest available stats for a team from features.csv.

    Searches for the team's most recent appearance as either home or away
    and extracts Elo, form, streak, and days-since-last-match data.
    """
    home_rows = df[df["home_team"] == team_name]
    away_rows = df[df["away_team"] == team_name]

    latest_home = home_rows.iloc[-1] if len(home_rows) > 0 else None
    latest_away = away_rows.iloc[-1] if len(away_rows) > 0 else None

    # Pick whichever appearance is more recent
    if latest_home is not None and latest_away is not None:
        if latest_home["date"] >= latest_away["date"]:
            latest = latest_home
            side = "home"
        else:
            latest = latest_away
            side = "away"
    elif latest_home is not None:
        latest = latest_home
        side = "home"
    elif latest_away is not None:
        latest = latest_away
        side = "away"
    else:
        return None

    # Elo: use home_elo_before or away_elo_before depending on which side
    elo = float(latest["home_elo_before"] if side == "home" else latest["away_elo_before"])

    # Form: win rate over last 5 matches
    win_rate_5 = float(latest[f"{side}_win_rate_5"])

    # Current streak
    streak = int(latest[f"{side}_streak"])

    # Days since last match
    days_since = float(latest[f"{side}_days_since_last"])

    # Home advantage (only meaningful when the team was home)
    home_advantage = float(latest["home_advantage"]) if side == "home" else None

    # Last match date
    last_date = latest["date"]

    return {
        "elo": elo,
        "win_rate_5": win_rate_5,
        "streak": streak,
        "days_since_last": days_since,
        "home_advantage": home_advantage,
        "last_date": last_date,
        "side": side,
    }


def _streak_label(streak: int) -> str:
    """Convert a numeric streak to a human-readable label."""
    if streak > 0:
        return f"W{streak}"
    elif streak < 0:
        return f"L{abs(streak)}"
    return "D"


def _streak_delta_color(streak: int) -> str:
    """Return Streamlit delta color keyword based on streak sign."""
    if streak > 0:
        return "normal"
    elif streak < 0:
        return "inverse"
    return "off"


def render_team_stats(df: pd.DataFrame, home_team: str, away_team: str):
    """Render pre-prediction stat cards for both teams side by side."""
    st.subheader("Team Overview")

    home_stats = get_team_latest_stats(df, home_team)
    away_stats = get_team_latest_stats(df, away_team)

    if home_stats is None and away_stats is None:
        st.info("No historical data found for the selected teams.")
        return

    col_home, col_away = st.columns(2)

    # --- Home team ---
    with col_home:
        st.markdown(
            f'<div style="text-align:center; padding:6px 0; font-size:1.2rem; '
            f'font-weight:600; border-bottom:3px solid #2e7d32;">{home_team}</div>',
            unsafe_allow_html=True,
        )
        if home_stats is None:
            st.info(f"No data available for {home_team}.")
        else:
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Elo Rating", f"{home_stats['elo']:.0f}")
            r1c2.metric(
                "Form (Last 5)",
                f"{home_stats['win_rate_5'] * 100:.0f}%",
            )
            r2c1, r2c2 = st.columns(2)
            r2c1.metric(
                "Current Streak",
                _streak_label(home_stats["streak"]),
                delta=f"{home_stats['streak']:+d}" if home_stats["streak"] != 0 else "0",
                delta_color=_streak_delta_color(home_stats["streak"]),
            )
            r2c2.metric("Days Since Last Match", f"{home_stats['days_since_last']:.0f}")
            if home_stats["home_advantage"] is not None:
                st.metric(
                    "Home Advantage",
                    f"{home_stats['home_advantage']:.1f}",
                    help="Historical home-field strength factor for this team",
                )

    # --- Away team ---
    with col_away:
        st.markdown(
            f'<div style="text-align:center; padding:6px 0; font-size:1.2rem; '
            f'font-weight:600; border-bottom:3px solid #c62828;">{away_team}</div>',
            unsafe_allow_html=True,
        )
        if away_stats is None:
            st.info(f"No data available for {away_team}.")
        else:
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Elo Rating", f"{away_stats['elo']:.0f}")
            r1c2.metric(
                "Form (Last 5)",
                f"{away_stats['win_rate_5'] * 100:.0f}%",
            )
            r2c1, r2c2 = st.columns(2)
            r2c1.metric(
                "Current Streak",
                _streak_label(away_stats["streak"]),
                delta=f"{away_stats['streak']:+d}" if away_stats["streak"] != 0 else "0",
                delta_color=_streak_delta_color(away_stats["streak"]),
            )
            r2c2.metric("Days Since Last Match", f"{away_stats['days_since_last']:.0f}")
