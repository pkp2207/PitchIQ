import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to sys.path so we can import src modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.predict import MatchPredictor

st.set_page_config(
    page_title="PitchIQ | FIFA Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    try:
        return MatchPredictor()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

predictor = load_predictor()

# Load team list
@st.cache_data
def load_teams():
    try:
        df = pd.read_csv(project_root / "data" / "processed" / "features.csv")
        teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
        return teams
    except:
        return ["Brazil", "Argentina", "France", "Germany", "England"]

teams = load_teams()

# --- Sidebar ---
with st.sidebar:
    st.title("⚽ PitchIQ")
    st.markdown("### FIFA World Cup Match Outcome Predictor")
    st.markdown("Predicts the outcome of international football matches using historical data, form, and Elo ratings.")
    st.divider()
    st.markdown("**Created By:** Param")
    st.markdown("**Data Source:** Kaggle")

# --- Main App ---
st.title("Match Prediction")

# Input section
col1, col2, col3 = st.columns(3)

with col1:
    home_team = st.selectbox("Home Team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)

with col2:
    away_team = st.selectbox("Away Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

with col3:
    tournament_type = st.selectbox("Tournament Type", ["Friendly", "Competitive", "FIFA World Cup"])

if home_team == away_team:
    st.warning("Please select different teams for Home and Away.")
    st.stop()

if st.button("Predict Outcome", type="primary", use_container_width=True):
    if predictor:
        with st.spinner("Analyzing historical data and computing features..."):
            try:
                prediction = predictor.predict_match(home_team, away_team, tournament_type)
                
                st.divider()
                st.subheader(f"Prediction: {prediction['outcome']} (for {home_team})")
                
                # Probabilities Chart
                probs = prediction['probabilities']
                labels = ['Win', 'Draw', 'Loss']
                values = [probs.get('Win', 0), probs.get('Draw', 0), probs.get('Loss', 0)]
                colors = ['#2e7d32', '#f9a825', '#c62828']
                
                fig = go.Figure(data=[go.Bar(
                    x=values,
                    y=['Probability'],
                    orientation='h',
                    text=[f"{v*100:.1f}%" for v in values],
                    textposition='auto',
                    marker_color=colors
                )])
                fig.update_layout(
                    barmode='stack',
                    title='Outcome Probabilities',
                    xaxis=dict(range=[0, 1], tickformat='.0%'),
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Attributions
                st.subheader("Top Contributing Features")
                for f in prediction['top_features']:
                    st.markdown(f"- **{f['feature']}**: {f['importance']:.4f}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
