"""Static mapping of assessment metrics to regulatory/standards references."""

STANDARDS_MAP = {
    # -------------------------------------------------------------------------
    # ACCOUNTABILITY — performance, robustness, privacy, data quality
    # -------------------------------------------------------------------------
    "accuracy_score": {
        "display": "Accuracy",
        "category": "Accountability",
        "eu_ai_act": "Art. 15 (Accuracy & robustness)",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Fraction of correct predictions.",
    },
    "roc_auc_score": {
        "display": "ROC-AUC",
        "category": "Accountability",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Area under the ROC curve — overall discriminative ability.",
    },
    "precision_score": {
        "display": "Precision",
        "category": "Accountability",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Fraction of positive predictions that are correct.",
    },
    "recall_score": {
        "display": "Recall (Sensitivity)",
        "category": "Accountability",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Fraction of actual positives correctly identified.",
    },
    "f1_score": {
        "display": "F1 Score",
        "category": "Accountability",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Harmonic mean of precision and recall.",
    },
    "membership_inference_attack_score": {
        "display": "Membership Inference Risk",
        "category": "Accountability",
        "eu_ai_act": "Art. 9 (Risk management) + GDPR Art. 25",
        "nist_ai_rmf": "MEASURE 2.6 / MANAGE 2.4",
        "iso_42001": "Annex A.7 (Privacy)",
        "description": "Attacker advantage in inferring training set membership.",
    },
    "population_stability_index": {
        "display": "Population Stability Index",
        "category": "Accountability",
        "eu_ai_act": "Art. 9 (Risk management) + Art. 17",
        "nist_ai_rmf": "MEASURE 2.7",
        "iso_42001": "Clause 9.1",
        "description": "Measures distributional shift between reference and current data.",
    },
    "adversarial_attack_success_rate": {
        "display": "Adversarial Attack Success Rate",
        "category": "Accountability",
        "eu_ai_act": "Art. 15 (Robustness & cybersecurity)",
        "nist_ai_rmf": "MEASURE 2.8",
        "iso_42001": "Annex A.8 (Security)",
        "description": "Fraction of adversarial inputs that fool the model.",
    },
    "class_imbalance_ratio": {
        "display": "Class Imbalance Ratio",
        "category": "Accountability",
        "eu_ai_act": "Art. 10 (Data governance)",
        "nist_ai_rmf": "MAP 1.5",
        "iso_42001": "Clause 8.2 (Data quality)",
        "description": "Ratio of minority to majority class in the assessment data.",
    },
    "feature_label_correlation": {
        "display": "Feature–Label Correlation",
        "category": "Accountability",
        "eu_ai_act": "Art. 10 (Data governance)",
        "nist_ai_rmf": "MAP 1.5",
        "iso_42001": "Clause 8.2 (Data quality)",
        "description": "Statistical association between each feature and the target label.",
    },
    "mixed_data_types": {
        "display": "Mixed Data Types Check",
        "category": "Accountability",
        "eu_ai_act": "Art. 10 (Data governance)",
        "nist_ai_rmf": "MAP 1.5",
        "iso_42001": "Clause 8.2 (Data quality)",
        "description": "Detects columns that contain a mix of data types, indicating data quality issues.",
    },

    # -------------------------------------------------------------------------
    # FAIRNESS — equitable outcomes across demographic groups
    # -------------------------------------------------------------------------
    "demographic_parity_difference": {
        "display": "Demographic Parity Difference",
        "category": "Fairness",
        "eu_ai_act": "Art. 10 (Data governance) + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2 / GOVERN 6.2",
        "iso_42001": "Annex A.6 (Fairness)",
        "description": "Difference in positive prediction rates across demographic groups.",
    },
    "equalized_odds_difference": {
        "display": "Equalized Odds Difference",
        "category": "Fairness",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Max difference in TPR and FPR across groups.",
    },
    "equal_opportunity_difference": {
        "display": "Equal Opportunity Difference",
        "category": "Fairness",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Difference in true positive rates across groups.",
    },
    "average_odds_difference": {
        "display": "Average Odds Difference",
        "category": "Fairness",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Average of TPR and FPR differences across groups.",
    },
    "disparate_impact_ratio": {
        "display": "Disparate Impact Ratio",
        "category": "Fairness",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Ratio of positive prediction rates — the 4/5ths rule threshold.",
    },
    "positive_label_rate": {
        "display": "Positive Outcome Rate by Group",
        "category": "Fairness",
        "eu_ai_act": "Art. 10 (Data governance) + Art. 5(1)(d)",
        "nist_ai_rmf": "MAP 1.5 / MEASURE 2.2",
        "iso_42001": "Annex A.6 (Fairness)",
        "description": "Rate at which each demographic group receives the positive label in the assessment data.",
    },

    # -------------------------------------------------------------------------
    # TRANSPARENCY — explainability and interpretability
    # -------------------------------------------------------------------------
    "shap_feature_importance": {
        "display": "SHAP Feature Importance",
        "category": "Transparency",
        "eu_ai_act": "Art. 13 (Transparency)",
        "nist_ai_rmf": "GOVERN 1.7 / MEASURE 2.9",
        "iso_42001": "Clause 8.4 (Transparency)",
        "description": "SHAP values indicating each feature's contribution to predictions.",
    },
}


def get_mapping(metric_key: str) -> dict:
    """Return the standards mapping for a given metric key, or an empty dict."""
    return STANDARDS_MAP.get(metric_key, {})


def get_all_mappings() -> dict:
    """Return the full standards map."""
    return STANDARDS_MAP
