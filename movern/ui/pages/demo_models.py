"""Demo Models assessment page."""

import streamlit as st

from movern.datasets.demo_models import AVAILABLE_MODELS, load_demo_model
from movern.ui._runner import run_assessment
from movern.ui.components.evaluator_selector import evaluator_selector
from movern.ui.components.results_display import display_results
from movern.ui.components.standards_panel import standards_panel
from movern.ui.report.docx_builder import build_report

MODEL_LABELS = {
    "credit": "Credit Risk — Fairness (age, sex), privacy, explainability",
    "hiring": "Hiring Screener — Fairness (gender, race), data fairness",
    "healthcare": "Healthcare Outcome — Fairness (age, race, sex), privacy",
    "fraud": "Fraud Detection — Performance, robustness, feature drift",
}


def render():
    st.title("Demo Model Assessment")
    st.markdown(
        "Select a pre-trained demo model, choose evaluators, and run an AI governance assessment."
    )

    # --- Model selection ---
    model_key = st.selectbox(
        "Select a demo model",
        options=AVAILABLE_MODELS,
        format_func=lambda k: MODEL_LABELS.get(k, k),
    )

    # --- Load model info ---
    try:
        pipeline, X_test, y_test, sensitive, metadata = load_demo_model(model_key)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # --- Model info card ---
    with st.expander("Model Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {metadata['name']}")
            st.markdown(f"**Description:** {metadata['description']}")
            st.markdown(f"**Target variable:** {metadata['target']}")
        with col2:
            st.markdown(f"**EU AI Act Risk Tier:** {metadata['eu_ai_act_risk']}")
            st.markdown(f"**Applicable Articles:** {', '.join(metadata['eu_ai_act_articles'])}")
            st.markdown(f"**Sensitive features:** {', '.join(metadata['sensitive_features']) or 'None'}")
        st.markdown(f"**Assessment set size:** {len(X_test)} rows × {len(X_test.columns)} features")

    # --- Evaluator selection ---
    selected_evaluators = evaluator_selector(model_key)

    # --- Run button ---
    if st.button("Run Assessment", type="primary", disabled=not selected_evaluators):
        with st.spinner("Running assessment…"):
            try:
                results = run_assessment(
                    pipeline, X_test, y_test, sensitive,
                    selected_evaluators, metadata,
                )
                st.session_state[f"results_{model_key}"] = results
                st.session_state[f"metadata_{model_key}"] = metadata
                st.session_state[f"evaluators_{model_key}"] = selected_evaluators
                st.success(f"Assessment complete — {len(results)} metric results.")
            except Exception as e:
                st.error(f"Assessment failed: {e}")
                return

    # --- Display results ---
    results = st.session_state.get(f"results_{model_key}")
    if results:
        st.divider()
        st.subheader("Results")
        display_results(results)
        st.divider()
        standards_panel(results)
        st.divider()

        # --- Report download ---
        st.subheader("Download Audit Report")
        try:
            report_bytes = build_report(
                results,
                st.session_state[f"metadata_{model_key}"],
                st.session_state[f"evaluators_{model_key}"],
            )
            st.download_button(
                label="Download Audit Report (.docx)",
                data=report_bytes,
                file_name=f"movern_audit_{model_key}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except ImportError as e:
            st.warning(f"Report generation unavailable: {e}")
