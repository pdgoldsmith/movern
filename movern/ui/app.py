"""
Movern Streamlit UI — main entry point.

Launch with:
    movern ui
    # or directly:
    streamlit run movern/ui/app.py
"""

import subprocess
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Auto-install required packages that are not part of the base install.
# Keys are the Python import name; values are the pip install spec.
# ---------------------------------------------------------------------------
_REQUIRED = {
    # Core UI / report
    "reportlab":       "reportlab>=4.0.0",
    "ucimlrepo":       "ucimlrepo>=0.0.3",
    # Evaluator dependencies (lightweight — installed at startup)
    "fairlearn":       "fairlearn>=0.7.0",
    "shap":            "shap>=0.41.0",
    "lifelines":       "lifelines>=0.27.3",
    "ydata_profiling": "ydata-profiling>=4.0.0",
    # deepchecks and adversarial-robustness-toolbox are large (~hundreds of MB)
    # and are installed on-demand only when those evaluators are selected.
}

def _ensure_packages():
    missing = []
    for module, pip_spec in _REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_spec)
    if missing:
        with st.spinner(f"Installing required packages: {', '.join(missing)}…"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", *missing]
            )
        st.rerun()

_ensure_packages()

st.set_page_config(
    page_title="Movern — AI Governance Assessment",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from movern.ui._pages import demo_models, home, own_model  # noqa: E402 (after set_page_config)

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
