import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def _extract_team_elo_history(df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Extract (date, elo) pairs for every match a team appeared in.

    When the team is the home side we take ``home_elo_before``;
    when the team is the away side we take ``away_elo_before``.
    The result is sorted chronologically.
    """
    home_mask = df["home_team"] == team_name
    away_mask = df["away_team"] == team_name

    home_part = df.loc[home_mask, ["date", "home_elo_before"]].rename(
        columns={"home_elo_before": "elo"}
    )
    away_part = df.loc[away_mask, ["date", "away_elo_before"]].rename(
        columns={"away_elo_before": "elo"}
    )

    combined = pd.concat([home_part, away_part], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def render_elo_chart(df: pd.DataFrame, home_team: str, away_team: str):
    """Render a Plotly line chart comparing the Elo history of two teams."""
    st.subheader("Elo Rating History")

    home_elo = _extract_team_elo_history(df, home_team)
    away_elo = _extract_team_elo_history(df, away_team)

    if home_elo.empty and away_elo.empty:
        st.info("No Elo history available for the selected teams.")
        return

    fig = go.Figure()

    if not home_elo.empty:
        fig.add_trace(go.Scatter(
            x=home_elo["date"],
            y=home_elo["elo"],
            mode="lines",
            name=home_team,
            line=dict(color="#2e7d32", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Elo: %{y:.0f}<extra>" + home_team + "</extra>",
        ))

    if not away_elo.empty:
        fig.add_trace(go.Scatter(
            x=away_elo["date"],
            y=away_elo["elo"],
            mode="lines",
            name=away_team,
            line=dict(color="#c62828", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Elo: %{y:.0f}<extra>" + away_team + "</extra>",
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Elo Rating",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
