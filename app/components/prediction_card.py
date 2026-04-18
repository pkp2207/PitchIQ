import streamlit as st
import plotly.graph_objects as go


OUTCOME_COLORS = {
    'Win': '#2e7d32',
    'Draw': '#f9a825',
    'Loss': '#c62828',
}


def render_prediction_card(prediction: dict, home_team: str, away_team: str):
    """Renders the outcome badge and probability stacked bar."""
    outcome = prediction['outcome']
    probs = prediction['probabilities']
    color = OUTCOME_COLORS.get(outcome, '#666')

    st.markdown(
        f'<div style="text-align:center; padding:12px; border-radius:8px; '
        f'background:{color}; color:white; font-size:1.5rem; font-weight:bold;">'
        f'{home_team} vs {away_team}: {outcome}</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Prediction from {home_team}'s perspective")

    # Stacked horizontal bar
    labels = ['Win', 'Draw', 'Loss']
    values = [probs.get(lbl, 0) for lbl in labels]
    colors = [OUTCOME_COLORS[lbl] for lbl in labels]

    fig = go.Figure()
    for lbl, val, clr in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            x=[val], y=[''], orientation='h', name=lbl,
            text=f"{val*100:.1f}%", textposition='inside',
            marker_color=clr, hovertemplate=f"{lbl}: {val*100:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 1], tickformat='.0%', showgrid=False),
        yaxis=dict(visible=False),
        height=80, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
