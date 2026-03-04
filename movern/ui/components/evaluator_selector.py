"""Reusable evaluator selection widget."""

from typing import Dict, List

import streamlit as st

EVALUATOR_DESCRIPTIONS: Dict[str, str] = {
    "Performance": "Accuracy, AUC, precision, recall, F1",
    "ModelFairness": "Demographic parity, equalized odds, equal opportunity",
    "DataFairness": "Statistical analysis of training data distribution across groups",
    "Privacy": "Membership inference attack risk (requires ART package)",
    "ShapExplainer": "SHAP feature importance (requires shap package)",
    "DataProfiler": "Full dataset profiling report (requires ydata-profiling)",
    "DeepChecks": "Comprehensive model and data integrity checks (requires deepchecks)",
    "FeatureDrift": "Population stability index for distribution shift detection",
}

DEFAULT_EVALUATORS: Dict[str, List[str]] = {
    "credit": ["Performance", "ModelFairness", "ShapExplainer"],
    "hiring": ["Performance", "ModelFairness", "DataFairness"],
    "healthcare": ["Performance", "ModelFairness"],
    "fraud": ["Performance", "FeatureDrift"],
    "custom": ["Performance"],
}


def evaluator_selector(model_key: str = "custom") -> List[str]:
    """Render an evaluator selection widget and return chosen evaluator names.

    Parameters
    ----------
    model_key : str
        Used to pre-select sensible defaults per demo model.

    Returns
    -------
    list of str
        Names of selected evaluators.
    """
    defaults = DEFAULT_EVALUATORS.get(model_key, DEFAULT_EVALUATORS["custom"])

    st.markdown("#### Select Evaluators")
    selected = []
    for name, description in EVALUATOR_DESCRIPTIONS.items():
        checked = st.checkbox(
            f"**{name}** — {description}",
            value=(name in defaults),
            key=f"ev_{model_key}_{name}",
        )
        if checked:
            selected.append(name)

    if not selected:
        st.warning("Select at least one evaluator.")

    return selected
