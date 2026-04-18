import streamlit as st
import plotly.graph_objects as go


def render_feature_attribution(top_features: list):
    """Renders a horizontal bar chart of SHAP values.

    Positive values (teal) push toward the predicted class.
    Negative values (orange) push away from the predicted class.
    """
    if not top_features:
        st.info("No feature attribution data available.")
        return

    # Sort by absolute value ascending so the largest bar is on top
    sorted_feats = sorted(top_features, key=lambda f: abs(f['shap_value']))
    names = [f['feature'].replace('_', ' ').title() for f in sorted_feats]
    values = [f['shap_value'] for f in sorted_feats]
    colors = ['#009688' if v >= 0 else '#ff7043' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='outside',
    ))

    fig.update_layout(
        title='Feature Attribution (SHAP)',
        xaxis_title='SHAP Value',
        yaxis=dict(automargin=True),
        height=max(200, 50 * len(sorted_feats)),
        margin=dict(l=0, r=40, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Teal = pushes toward predicted outcome | "
        "Orange = pushes against predicted outcome"
    )
