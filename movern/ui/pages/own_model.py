"""Own Model upload and assessment page."""

import io
import pickle

import pandas as pd
import streamlit as st

from movern.ui._runner import run_assessment
from movern.ui.components.evaluator_selector import evaluator_selector
from movern.ui.components.results_display import display_results
from movern.ui.components.standards_panel import standards_panel
from movern.ui.report.docx_builder import build_report


def render():
    st.title("Assess Your Own Model")
    st.markdown(
        "Upload a trained sklearn model (`.pkl`) and a labelled assessment CSV to run an "
        "AI governance assessment."
    )

    # --- File uploads ---
    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("Upload model (.pkl)", type=["pkl"])
    with col2:
        data_file = st.file_uploader("Upload assessment data (.csv)", type=["csv"])

    if not model_file or not data_file:
        st.info("Upload a model and assessment dataset to continue.")
        return

    # --- Load model ---
    try:
        pipeline = pickle.load(io.BytesIO(model_file.read()))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # --- Load data ---
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(), use_container_width=True)

    # --- Configuration ---
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Target (label) column", options=df.columns.tolist())
    with col2:
        sensitive_cols = st.multiselect(
            "Sensitive feature columns (optional)",
            options=[c for c in df.columns if c != target_col],
        )

    model_name = st.text_input("Model name (for report)", value="My Model")
    model_description = st.text_input("Model description (for report)", value="")

    # --- Build feature matrix ---
    feature_cols = [c for c in df.columns if c != target_col]
    X_test = df[feature_cols]
    y_test = df[target_col]
    sensitive = df[sensitive_cols] if sensitive_cols else pd.DataFrame()

    metadata = {
        "name": model_name,
        "description": model_description,
        "features": feature_cols,
        "sensitive_features": sensitive_cols,
        "target": target_col,
        "eu_ai_act_risk": "Not assessed",
        "eu_ai_act_articles": [],
    }

    # --- Evaluator selection ---
    selected_evaluators = evaluator_selector("custom")

    # --- Run ---
    if st.button("Run Assessment", type="primary", disabled=not selected_evaluators):
        with st.spinner("Running assessment…"):
            try:
                results = run_assessment(
                    pipeline, X_test, y_test,
                    sensitive if not sensitive.empty else None,
                    selected_evaluators, metadata,
                )
                st.session_state["own_results"] = results
                st.session_state["own_metadata"] = metadata
                st.session_state["own_evaluators"] = selected_evaluators
                st.success(f"Assessment complete — {len(results)} metric results.")
            except Exception as e:
                st.error(f"Assessment failed: {e}")
                return

    # --- Display results ---
    results = st.session_state.get("own_results")
    if results:
        st.divider()
        st.subheader("Results")
        display_results(results)
        st.divider()
        standards_panel(results)
        st.divider()

        st.subheader("Download Audit Report")
        try:
            report_bytes = build_report(
                results,
                st.session_state["own_metadata"],
                st.session_state["own_evaluators"],
            )
            st.download_button(
                label="Download Audit Report (.docx)",
                data=report_bytes,
                file_name="movern_audit_custom.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except ImportError as e:
            st.warning(f"Report generation unavailable: {e}")
