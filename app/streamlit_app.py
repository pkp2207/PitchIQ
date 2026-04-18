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
def load_teams():
    try:
        df = pd.read_csv(project_root / "data" / "processed" / "features.csv")
        teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        return teams
    except Exception:
        return ["Brazil", "Argentina", "France", "Germany", "England"]


predictor = load_predictor()
teams = load_teams()

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

if st.button("Predict Outcome", type="primary", use_container_width=True):
    if predictor:
        with st.spinner("Analyzing historical data and computing features..."):
            try:
                prediction = predictor.predict_match(home_team, away_team, tournament_type)

                st.divider()

                # 1. Prediction card with outcome badge + probability bar
                render_prediction_card(prediction, home_team, away_team)

                # 2. SHAP feature attribution chart
                st.subheader("Feature Attribution")
                render_feature_attribution(prediction['top_features'])

                # 3. Head-to-head history
                st.subheader("Head-to-Head History")
                h2h = predictor.get_h2h_history(home_team, away_team)
                render_h2h_table(h2h, home_team, away_team)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
