import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path so we can import src/app modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.predict import MatchPredictor
from app.components.prediction_card import render_prediction_card
from app.components.feature_attribution import render_feature_attribution
from app.components.h2h_table import render_h2h_table
from app.components.team_stats import render_team_stats
from app.components.elo_chart import render_elo_chart
from app.components.sentiment_card import render_sentiment_card

st.set_page_config(
    page_title="PitchIQ | FIFA Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor():
    try:
        return MatchPredictor()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_features_df():
    """Load features.csv once and cache it for teams, stats, and charts."""
    try:
        return pd.read_csv(project_root / "data" / "processed" / "features.csv")
    except Exception:
        return None


@st.cache_data
def load_teams(_df):
    if _df is not None:
        return sorted(set(_df['home_team'].unique()) | set(_df['away_team'].unique()))
    return ["Brazil", "Argentina", "France", "Germany", "England"]


@st.cache_data
def load_sentiment_data():
    """Load sentiment.csv and return a DataFrame."""
    try:
        path = project_root / "data" / "raw" / "sentiment.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, parse_dates=['date'])
    except Exception:
        return None


def get_team_sentiment(sentiment_df, team_name, before_date=None):
    """Get recent sentiment for a team (last 7 days of data or latest available)."""
    if sentiment_df is None:
        return {'avg_score': 0.0, 'volume': 0, 'headlines': []}
    team_data = sentiment_df[sentiment_df['team'] == team_name]
    if before_date:
        team_data = team_data[team_data['date'] < before_date]
    # Get last 7 days of data
    if len(team_data) > 0:
        latest_date = team_data['date'].max()
        week_ago = latest_date - pd.Timedelta(days=7)
        recent = team_data[team_data['date'] >= week_ago]
    else:
        recent = team_data
    return {
        'avg_score': float(recent['sentiment_score'].mean()) if len(recent) > 0 else 0.0,
        'volume': len(recent),
        'headlines': recent['headline'].tail(3).tolist() if 'headline' in recent.columns else [],
    }


predictor = load_predictor()
features_df = load_features_df()
teams = load_teams(features_df)
sentiment_df = load_sentiment_data()

# --- Sidebar ---
with st.sidebar:
    st.title("⚽ PitchIQ")
    st.markdown("### FIFA Match Outcome Predictor")
    st.markdown(
        "Predicts the outcome of international football matches "
        "using historical data, form, and Elo ratings."
    )
    st.divider()
    st.markdown("**Created By:** Param")
    st.markdown("**Data Source:** Kaggle")

# --- Main ---
st.title("Match Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    home_team = st.selectbox(
        "Home Team", teams,
        index=teams.index("Brazil") if "Brazil" in teams else 0,
    )

with col2:
    away_team = st.selectbox(
        "Away Team", teams,
        index=teams.index("Argentina") if "Argentina" in teams else 1,
    )

with col3:
    tournament_type = st.selectbox(
        "Tournament Type",
        ["Friendly", "Competitive", "FIFA World Cup"],
    )

if home_team == away_team:
    st.warning("Please select different teams for Home and Away.")
    st.stop()

# --- Team Stats Cards (pre-prediction context) ---
if features_df is not None:
    render_team_stats(features_df, home_team, away_team)
    st.divider()

# --- Sentiment Cards ---
if sentiment_df is not None:
    home_sentiment = get_team_sentiment(sentiment_df, home_team)
    away_sentiment = get_team_sentiment(sentiment_df, away_team)
    render_sentiment_card(home_team, away_team, home_sentiment, away_sentiment)
    st.divider()

# Run prediction on button press and store in session state
if st.button("Predict Outcome", type="primary", use_container_width=True):
    if predictor:
        with st.spinner("Analyzing historical data and computing features..."):
            try:
                prediction = predictor.predict_match(home_team, away_team, tournament_type)
                st.session_state['prediction'] = prediction
                st.session_state['pred_home'] = home_team
                st.session_state['pred_away'] = away_team
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.session_state.pop('prediction', None)

# Display results if a prediction exists in session state
if 'prediction' in st.session_state:
    prediction = st.session_state['prediction']
    pred_home = st.session_state['pred_home']
    pred_away = st.session_state['pred_away']

    st.divider()

    # 1. Prediction card with outcome badge + probability bar
    render_prediction_card(prediction, pred_home, pred_away)

    # 2. SHAP feature attribution chart
    st.subheader("Feature Attribution")
    render_feature_attribution(prediction['top_features'])

    # 3. Head-to-head history
    st.subheader("Head-to-Head History")
    h2h_count = st.slider(
        "Number of past meetings to show",
        min_value=5, max_value=50, value=10, step=5,
    )
    h2h = predictor.get_h2h_history(pred_home, pred_away, last_n=h2h_count)
    render_h2h_table(h2h, pred_home, pred_away)

    # 4. Elo rating history chart
    if features_df is not None:
        st.divider()
        render_elo_chart(features_df, pred_home, pred_away)
