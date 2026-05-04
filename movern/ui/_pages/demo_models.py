"""Demo Models assessment page."""

import streamlit as st

from movern.datasets.demo_models import AVAILABLE_MODELS, load_demo_model
from movern.ui._runner import get_packages_to_install, install_packages, run_assessment
from movern.ui.components.evaluator_selector import evaluator_selector
from movern.ui.components.results_display import display_results
from movern.ui.components.standards_panel import standards_panel
from movern.ui.report.pdf_builder import build_report

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

    # --- Load model (auto-trains on first use) ---
    try:
        with st.spinner(f"Loading {MODEL_LABELS.get(model_key, model_key).split(' —')[0]} model… (first load may take a minute)"):
            pipeline, X_test, y_test, sensitive, X_train_sample, y_train_sample, metadata = load_demo_model(model_key)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # --- Dataset info card ---
    ds = metadata.get("dataset", {})
    if ds:
        with st.expander("Dataset Source & Provenance", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{ds['name']}**")
                st.markdown(ds["description"])
                st.markdown(f"*{ds['citation']}*")
            with col2:
                st.markdown(f"**Source:** [{ds['source']}]({ds['url']})")
                st.markdown(f"**License:** {ds['license']}")
                st.markdown(f"**Full dataset:** {ds['n_rows']:,} rows × {ds['n_cols']} features")
                st.markdown(f"**Assessment set:** {len(X_test):,} rows")

    # --- Assessment data card ---
    ad = metadata.get("assessment_data", {})
    with st.expander("Assessment Data", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            if ad.get("description"):
                st.markdown(ad["description"])
            else:
                st.markdown(
                    f"Assessment set drawn from the source dataset above. "
                    f"Contains {len(X_test.columns)} features and "
                    f"{len(X_test):,} rows held out for evaluation."
                )
            if ad.get("sensitive_feature_notes"):
                st.markdown(f"**Sensitive features:** {ad['sensitive_feature_notes']}")
        with col2:
            st.markdown(f"**Rows:** {len(X_test):,}")
            st.markdown(f"**Features:** {len(X_test.columns)}")
            if ad.get("split"):
                st.markdown(f"**Split method:** {ad['split']}")
            # Class balance
            pos = int(y_test.sum())
            neg = int((y_test == 0).sum())
            total = len(y_test)
            pos_label = ad.get("positive_label", f"1 = {metadata['target']}")
            neg_label = ad.get("negative_label", f"0 = not {metadata['target']}")
            st.markdown(
                f"**Class balance:**  \n"
                f"{pos_label} → {pos:,} ({100*pos/total:.1f}%)  \n"
                f"{neg_label} → {neg:,} ({100*neg/total:.1f}%)"
            )
            if not sensitive.empty:
                st.markdown(f"**Sensitive columns:** {', '.join(sensitive.columns)}")

    # --- Model info card ---
    with st.expander("Model Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {metadata['name']}")
            st.markdown(f"**Description:** {metadata['description']}")
            st.markdown(f"**Target variable:** {metadata['target']}")
        with col2:
            st.markdown(f"**EU AI Act Risk Tier:** {metadata['eu_ai_act_risk']}")
            st.markdown(f"**Applicable Articles:** {', '.join(metadata['eu_ai_act_articles'])}")
            st.markdown(f"**Sensitive features:** {', '.join(metadata['sensitive_features']) or 'None'}")

    # --- Evaluator selection ---
    selected_evaluators = evaluator_selector(model_key)

    # --- Run button ---
    if st.button("Run Assessment", type="primary", disabled=not selected_evaluators):
        to_install = get_packages_to_install(selected_evaluators)
        if to_install:
            with st.spinner(f"Installing {', '.join(to_install)} (one-time, may take a minute)…"):
                install_packages(to_install)
        with st.spinner("Running assessment…"):
            try:
                results = run_assessment(
                    pipeline, X_test, y_test, sensitive,
                    selected_evaluators, metadata,
                    X_train=X_train_sample, y_train=y_train_sample,
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
                label="Download Audit Report (.pdf)",
                data=report_bytes,
                file_name=f"movern_audit_{model_key}.pdf",
                mime="application/pdf",
            )
        except ImportError as e:
            st.warning(f"Report generation unavailable: {e}")
