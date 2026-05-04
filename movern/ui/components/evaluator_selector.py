"""Reusable evaluator selection widget."""

from typing import Dict, List

import streamlit as st

EVALUATOR_METADATA: Dict[str, Dict[str, str]] = {
    "Performance": {
        "description": "Measures predictive accuracy using metrics like accuracy, AUC, precision, recall, and F1. Establishes the baseline capability of the model.",
        "eu_ai_act": "Art. 15 (Accuracy & robustness)",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
    },
    "ModelFairness": {
        "description": "Tests whether the model produces equitable outcomes across demographic groups using metrics like demographic parity difference and equalized odds.",
        "eu_ai_act": "Art. 10 (Data governance) + Art. 5(1)(d) (Bias prohibition)",
        "nist_ai_rmf": "MEASURE 2.2 / GOVERN 6.2",
        "iso_42001": "Annex A.6 (Fairness)",
    },
    "DataFairness": {
        "description": "Analyses the training data distribution across sensitive groups to detect representational bias before it propagates to model outcomes.",
        "eu_ai_act": "Art. 10 (Data governance)",
        "nist_ai_rmf": "MAP 1.5 / MEASURE 2.2",
        "iso_42001": "Annex A.6 (Fairness)",
    },
    "Privacy": {
        "description": "Runs membership inference attacks to estimate how much the model reveals about its training data. Requires the adversarial-robustness-toolbox package.",
        "eu_ai_act": "Art. 9 (Risk management) + GDPR Art. 25",
        "nist_ai_rmf": "MEASURE 2.6 / MANAGE 2.4",
        "iso_42001": "Annex A.7 (Privacy)",
    },
    "ShapExplainer": {
        "description": "Computes SHAP values to explain individual predictions and rank feature importance. Requires the shap package.",
        "eu_ai_act": "Art. 13 (Transparency & explainability)",
        "nist_ai_rmf": "GOVERN 1.7 / MEASURE 2.9",
        "iso_42001": "Clause 8.4 (Transparency)",
    },
    "DataProfiler": {
        "description": "Generates a full statistical profile of the assessment dataset including distributions, missing values, and correlations. Requires ydata-profiling.",
        "eu_ai_act": "Art. 10 (Data governance)",
        "nist_ai_rmf": "MAP 1.5",
        "iso_42001": "Clause 8.2 (Data quality)",
    },
    "DeepChecks": {
        "description": "Runs a comprehensive suite of data integrity and model validation checks. Requires the deepchecks package.",
        "eu_ai_act": "Art. 9 (Risk management) + Art. 15 (Robustness)",
        "nist_ai_rmf": "MEASURE 2.5 / MEASURE 2.7",
        "iso_42001": "Clause 9.1",
    },
    "FeatureDrift": {
        "description": "Detects distributional shift between reference and current data using Population Stability Index. Important for models deployed over time.",
        "eu_ai_act": "Art. 9 (Risk management) + Art. 17 (Post-market monitoring)",
        "nist_ai_rmf": "MEASURE 2.7",
        "iso_42001": "Clause 9.1",
    },
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
    st.divider()

    selected = []
    for name, meta in EVALUATOR_METADATA.items():
        checked = st.checkbox(f"**{name}**", value=(name in defaults), key=f"ev_{model_key}_{name}")
        st.caption(meta["description"])
        st.caption(
            f"🇪🇺 EU AI Act: {meta['eu_ai_act']}  ·  "
            f"🇺🇸 NIST AI RMF: {meta['nist_ai_rmf']}  ·  "
            f"🌐 ISO 42001: {meta['iso_42001']}"
        )
        st.markdown("")
        if checked:
            selected.append(name)

    st.divider()
    if not selected:
        st.warning("Select at least one evaluator.")

    return selected
