import streamlit as st
import pandas as pd


def render_h2h_table(h2h_records: list, home_team: str, away_team: str):
    """Renders the last N head-to-head encounters and a win/draw/loss tally."""
    if not h2h_records:
        st.info(f"No prior meetings found between {home_team} and {away_team}.")
        return

    rows = []
    home_wins = draws = away_wins = 0
    for rec in h2h_records:
        outcome_val = rec['outcome']
        rec_home = rec['home_team']

        # Determine result from home_team's perspective (the team selected as home)
        if rec_home == home_team:
            if outcome_val == 1:
                result = f"{home_team} Win"
                home_wins += 1
            elif outcome_val == 0:
                result = "Draw"
                draws += 1
            else:
                result = f"{away_team} Win"
                away_wins += 1
        else:
            # The teams were reversed in this historical match
            if outcome_val == 1:
                result = f"{away_team} Win"
                away_wins += 1
            elif outcome_val == 0:
                result = "Draw"
                draws += 1
            else:
                result = f"{home_team} Win"
                home_wins += 1

        rows.append({
            'Date': rec['date'],
            'Home': rec_home,
            'Away': rec['away_team'],
            'Result': result,
        })

    # Tally
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{home_team} Wins", home_wins)
    col2.metric("Draws", draws)
    col3.metric(f"{away_team} Wins", away_wins)

    # Table
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
