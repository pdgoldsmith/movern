"""
Movern Streamlit UI — main entry point.

Launch with:
    movern ui
    # or directly:
    streamlit run movern/ui/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Movern — AI Governance Assessment",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from movern.ui.pages import demo_models, home, own_model  # noqa: E402 (after set_page_config)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚖️ Movern")
    st.markdown("AI Model Governance Assessment")
    st.divider()

    page = st.radio(
        "Navigate",
        options=["Home", "Demo Models", "Own Model"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "<small>Movern is a fork of [Lens by Credo AI](https://github.com/credo-ai/credoai_lens).</small>",
        unsafe_allow_html=True,
    )
    st.markdown("<small>v0.1.0</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if page == "Home":
    home.render()
elif page == "Demo Models":
    demo_models.render()
elif page == "Own Model":
    own_model.render()
