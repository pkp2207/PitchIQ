import streamlit as st


# Thresholds for sentiment classification
_POSITIVE_THRESHOLD = 0.3
_NEGATIVE_THRESHOLD = -0.3

# Colour palette consistent with prediction_card.py
_SENTIMENT_STYLES = {
    "Positive": {"color": "#2e7d32", "emoji": "+"},
    "Neutral":  {"color": "#f9a825", "emoji": "~"},
    "Negative": {"color": "#c62828", "emoji": "-"},
}


def _classify_sentiment(score: float) -> str:
    """Map a numeric sentiment score to a human-readable label."""
    if score > _POSITIVE_THRESHOLD:
        return "Positive"
    elif score < _NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def _render_team_sentiment(team_name: str, sentiment: dict, border_color: str):
    """Render a single team's sentiment block inside a column.

    Parameters
    ----------
    team_name : str
        Display name of the team.
    sentiment : dict
        Keys: ``avg_score`` (float), ``volume`` (int), ``headlines`` (list[str]).
    border_color : str
        CSS colour for the header underline (green for home, red for away).
    """
    st.markdown(
        f'<div style="text-align:center; padding:6px 0; font-size:1.1rem; '
        f'font-weight:600; border-bottom:3px solid {border_color};">'
        f'{team_name}</div>',
        unsafe_allow_html=True,
    )

    score = sentiment.get("avg_score", 0.0)
    volume = sentiment.get("volume", 0)
    headlines = sentiment.get("headlines", [])

    label = _classify_sentiment(score)
    style = _SENTIMENT_STYLES[label]

    # Sentiment badge
    st.markdown(
        f'<div style="text-align:center; margin:8px 0; padding:4px 12px; '
        f'border-radius:6px; background:{style["color"]}; color:white; '
        f'font-weight:600; display:inline-block;">'
        f'{label}</div>',
        unsafe_allow_html=True,
    )

    # Metrics row
    m1, m2 = st.columns(2)
    m1.metric("Sentiment Score", f"{score:+.2f}")
    m2.metric("Media Volume", f"{volume}")

    # Recent headlines in an expander
    if headlines:
        with st.expander("Recent Headlines"):
            for headline in headlines[:3]:
                st.markdown(f"- {headline}")
    else:
        st.caption("No recent headlines available.")


def render_sentiment_card(
    home_team: str,
    away_team: str,
    home_sentiment: dict,
    away_sentiment: dict,
):
    """Display a side-by-side sentiment comparison for two teams.

    Parameters
    ----------
    home_team, away_team : str
        Team names.
    home_sentiment, away_sentiment : dict
        Each dict must contain:
        - ``avg_score`` (float): average sentiment score
        - ``volume`` (int): number of articles / posts analysed
        - ``headlines`` (list[str]): most recent headlines
    """
    st.subheader("Media Sentiment")

    col_home, col_away = st.columns(2)

    with col_home:
        _render_team_sentiment(home_team, home_sentiment, border_color="#2e7d32")

    with col_away:
        _render_team_sentiment(away_team, away_sentiment, border_color="#c62828")
