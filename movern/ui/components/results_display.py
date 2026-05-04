"""Render assessment results in the Streamlit UI."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Plain-English display labels for technical metric names.
METRIC_LABELS: Dict[str, str] = {
    # Performance
    "accuracy_score": "Accuracy",
    "balanced_accuracy_score": "Balanced Accuracy",
    "roc_auc_score": "AUC (Discrimination Ability)",
    "average_precision_score": "Average Precision",
    "f1_score": "F1 Score (Precision-Recall Balance)",
    "precision_score": "Precision (Positive Predictive Value)",
    "true_positive_rate": "Recall / Sensitivity (True Positive Rate)",
    "true_negative_rate": "Specificity (True Negative Rate)",
    "false_positive_rate": "False Positive Rate",
    "false_negative_rate": "False Negative Rate (Miss Rate)",
    "false_discovery_rate": "False Discovery Rate",
    "false_omission_rate": "False Omission Rate",
    "gini_coefficient": "Gini Coefficient (Discriminatory Power)",
    "matthews_correlation_coefficient": "Matthews Correlation Coefficient",
    "selection_rate": "Selection Rate (Positive Rate)",
    "overprediction": "Overprediction",
    "underprediction": "Underprediction",
    "ks_score_binary": "KS Statistic",
    # Regression
    "mean_absolute_error": "Mean Absolute Error (MAE)",
    "mean_squared_error": "Mean Squared Error (MSE)",
    "root_mean_squared_error": "Root Mean Squared Error (RMSE)",
    "mean_absolute_percentage_error": "Mean Absolute Percentage Error (MAPE)",
    "mean_squared_log_error": "Mean Squared Log Error",
    "median_absolute_error": "Median Absolute Error",
    "r2_score": "R² (Goodness of Fit)",
    "explained_variance_score": "Explained Variance",
    "max_error": "Maximum Error",
    "target_ks_statistic": "KS Statistic (Regression)",
    # Fairness
    "demographic_parity_difference": "Demographic Parity Difference (Selection Rate Gap)",
    "demographic_parity_ratio": "Demographic Parity Ratio (Adverse Impact Ratio)",
    "equalized_odds_difference": "Equalized Odds Difference",
    "equal_opportunity_difference": "Equal Opportunity Difference (True Positive Rate Gap)",
    "average_odds_difference": "Average Odds Difference",
    # Data fairness
    "positive_label_rate": "Positive Outcome Rate by Group",
    "sensitive_feature_prediction_score": "Sensitive Feature Predictability (Proxy Risk)",
    "max_proxy_mutual_information": "Maximum Proxy Mutual Information",
    # Privacy
    "membership_inference_attack_score": "Membership Inference Attack Accuracy",
    "attacker_advantage": "Attacker Advantage (Privacy Leakage)",
    "model_based_attack_score": "Model-Based Privacy Attack Score",
    "rule_based_attack_score": "Rule-Based Privacy Attack Score",
    # Drift
    "population_stability_index": "Population Stability Index (Distribution Shift)",
    # DeepChecks
    "class_imbalance_ratio": "Class Imbalance Ratio",
    "feature_label_correlation": "Feature–Label Correlation",
    "mixed_data_types": "Mixed Data Types Check",
}


def _label(metric: str) -> str:
    """Return a plain-English label for a metric, falling back to the raw name."""
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


# Acceptable-range guidance per metric name.
# "range" is a human-readable description; "note" cites the source/framework.
METRIC_GUIDANCE: Dict[str, Dict[str, str]] = {
    # --- Performance ---
    "accuracy_score": {
        "range": "Higher is better. Domain-dependent; > 0.80 is a common minimum for high-stakes classifiers.",
        "note": "No universal threshold. Justify against baseline/random rates per EU AI Act Art. 15.",
    },
    "balanced_accuracy_score": {
        "range": "0–1, higher is better. Preferred over accuracy on imbalanced datasets.",
        "note": "No standard threshold; compare to random-classifier baseline (0.5 for binary).",
    },
    "roc_auc_score": {
        "range": "0.5 (random) – 1.0 (perfect). > 0.80 is generally considered good.",
        "note": "Widely used but no regulatory floor. Document chosen threshold per EU AI Act Art. 15.",
    },
    "f1_score": {
        "range": "0–1, higher is better. Balances precision and recall.",
        "note": "No standard threshold. Useful when false negatives and false positives have similar cost.",
    },
    "precision_score": {
        "range": "0–1, higher is better. Prioritise when false positives are costly.",
        "note": "No regulatory floor. Interpret alongside recall.",
    },
    "true_positive_rate": {
        "range": "0–1, higher is better (also called Recall / Sensitivity).",
        "note": "In medical/safety contexts regulators often require > 0.90. Document justification.",
    },
    "false_positive_rate": {
        "range": "0–1, lower is better.",
        "note": "No standard threshold. High FPR may indicate bias against a group.",
    },
    "false_negative_rate": {
        "range": "0–1, lower is better.",
        "note": "In high-stakes domains (healthcare, fraud) keep as low as possible.",
    },
    "average_precision_score": {
        "range": "0–1, higher is better. Summarises the precision-recall curve.",
        "note": "No regulatory threshold. Robust to class imbalance.",
    },
    "gini_coefficient": {
        "range": "0–1, higher is better (discriminatory power).",
        "note": "Common in credit scoring. > 0.40 considered acceptable by many lenders.",
    },
    "matthews_correlation_coefficient": {
        "range": "-1 to +1. Closer to +1 is better; 0 = random; negative = worse than random.",
        "note": "No standard threshold. Robust single-number summary for imbalanced problems.",
    },
    "population_stability_index": {
        "range": "< 0.10 stable · 0.10–0.25 slight shift (monitor) · > 0.25 significant shift (investigate).",
        "note": "Rule-of-thumb thresholds from credit-risk industry practice. Cited in NIST AI RMF MEASURE 2.7.",
    },
    "class_imbalance_ratio": {
        "range": "Ideal: balanced classes (ratio near 1.0). Ratios below 0.1 indicate severe imbalance.",
        "note": "Severe imbalance can cause a model to ignore the minority class. Consider resampling or adjusted thresholds. Relevant to EU AI Act Art. 10 (data governance).",
    },
    "feature_label_correlation": {
        "range": "0–1. Higher values indicate stronger association with the target.",
        "note": "Very high correlation (> 0.9) may indicate data leakage. High correlation with a sensitive feature warrants a proxy discrimination review.",
    },
    "mixed_data_types": {
        "range": "Pass = no mixed types detected.",
        "note": "Mixed types in a column (e.g. numbers stored as strings) can silently degrade model performance and are a data quality concern under EU AI Act Art. 10.",
    },
    # --- Fairness ---
    "demographic_parity_difference": {
        "range": "Ideal = 0. |value| < 0.10 is a common rule of thumb for acceptability.",
        "note": "NIST SP 1270 (bias guidance) suggests < 0.10. EU AI Act Art. 5(1)(d) / Art. 10 require bias to be 'addressed'.",
    },
    "demographic_parity_ratio": {
        "range": "Ideal = 1.0. ≥ 0.80 is the US '80% rule' (adverse impact ratio).",
        "note": "Originates from US EEOC employment law. Adopted informally in EU fairness assessments.",
    },
    "equalized_odds_difference": {
        "range": "Ideal = 0. |value| < 0.10 is a common rule of thumb.",
        "note": "No regulatory mandate, but frequently cited in academic literature and NIST SP 1270.",
    },
    "equal_opportunity_difference": {
        "range": "Ideal = 0. |value| < 0.10 is a common rule of thumb.",
        "note": "Focuses on true positive rate parity. Important in benefit-allocation contexts.",
    },
    # --- Privacy ---
    "membership_inference_attack_score": {
        "range": "0.50 = no leakage (random guess). Closer to 1.0 = higher privacy risk.",
        "note": "No regulatory numeric threshold. GDPR Art. 25 / EU AI Act Art. 9 require risk to be 'minimised'. Values > 0.60 warrant investigation.",
    },
    "attacker_advantage": {
        "range": "0.0 = no advantage (model is private). 1.0 = perfect membership inference.",
        "note": "Normalised measure of how much better than random the attack performs. Values < 0.1 are generally considered acceptable.",
    },
    "model_based_attack_score": {
        "range": "0.50 = no leakage. Higher values indicate greater membership inference risk.",
        "note": "Same framework as membership_inference_attack_score.",
    },
    "rule_based_attack_score": {
        "range": "0.50 = no leakage. Higher values indicate greater membership inference risk.",
        "note": "Same framework as membership_inference_attack_score.",
    },
    # --- Data fairness ---
    "positive_label_rate": {
        "range": "Ideal: equal rates across all groups. Large differences indicate representational bias in the data.",
        "note": "No universal threshold. The US '80% rule' suggests the lowest group rate should be ≥ 80% of the highest. Disparities here reflect bias in the training data before the model is even applied. Relevant to EU AI Act Art. 10 (data governance).",
    },
    "sensitive_feature_prediction_score": {
        "range": "Ideal = 0.50 (unpredictable from features). Higher values suggest proxy discrimination risk.",
        "note": "No standard threshold. Values approaching random classifier (0.5) are preferred.",
    },
    "max_proxy_mutual_information": {
        "range": "Lower is better. Values near 0 indicate low proxy risk.",
        "note": "No standard threshold. Elevated values suggest a feature acts as a proxy for a sensitive attribute.",
    },
}


def display_results(results: List[Dict[str, Any]]) -> None:
    """Render a list of metric result dicts as expandable tables + bar charts.

    Parameters
    ----------
    results : list of dict
        Each dict: ``{"evaluator": str, "metric": str, "value": float|str, "group": str}``
    """
    if not results:
        st.info("No results to display.")
        return

    # Group by evaluator
    by_evaluator: Dict[str, List] = {}
    for r in results:
        ev = r.get("evaluator", "General")
        by_evaluator.setdefault(ev, []).append(r)

    for ev_name, ev_results in by_evaluator.items():
        with st.expander(f"**{ev_name}**", expanded=True):
            df = pd.DataFrame(ev_results)
            cols_to_show = [c for c in ["metric", "group", "value"] if c in df.columns]
            df_display = df[cols_to_show].copy()
            if "metric" in df_display.columns:
                df_display["metric"] = df_display["metric"].apply(_label)
            if "value" in df_display.columns:
                df_display["value"] = df_display["value"].apply(
                    lambda v: f"{v:.4f}" if isinstance(v, float) else v
                )
            df_display = df_display.rename(columns={"metric": "Metric", "group": "Group", "value": "Value"})
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Bar chart for numeric scalar metrics (no group breakdown)
            numeric_rows = [
                r for r in ev_results
                if isinstance(r.get("value"), (int, float))
                and r.get("group", "Overall") == "Overall"
            ]
            if numeric_rows:
                chart_df = (
                    pd.DataFrame(numeric_rows)
                    .assign(metric=lambda d: d["metric"].apply(_label))
                    .set_index("metric")["value"]
                )
                st.bar_chart(chart_df)

            # Interpretation guide — only for metrics that have guidance entries
            present_metrics = [r.get("metric") for r in ev_results]
            guided = {m: METRIC_GUIDANCE[m] for m in present_metrics if m in METRIC_GUIDANCE}
            if guided:
                with st.expander("ℹ️ Interpretation guide", expanded=False):
                    guide_rows = [
                        {
                            "Metric": _label(m),
                            "Acceptable range": g["range"],
                            "Framework context": g["note"],
                        }
                        for m, g in guided.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(guide_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
