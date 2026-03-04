"""Static mapping of assessment metrics to regulatory/standards references."""

STANDARDS_MAP = {
    # Performance metrics
    "accuracy_score": {
        "display": "Accuracy",
        "eu_ai_act": "Art. 15 (Accuracy & robustness)",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Fraction of correct predictions.",
    },
    "roc_auc_score": {
        "display": "ROC-AUC",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Area under the ROC curve — overall discriminative ability.",
    },
    "precision_score": {
        "display": "Precision",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Fraction of positive predictions that are correct.",
    },
    "recall_score": {
        "display": "Recall (Sensitivity)",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Fraction of actual positives correctly identified.",
    },
    "f1_score": {
        "display": "F1 Score",
        "eu_ai_act": "Art. 15",
        "nist_ai_rmf": "MEASURE 2.5",
        "iso_42001": "Clause 9.1",
        "description": "Harmonic mean of precision and recall.",
    },
    # Fairness metrics
    "demographic_parity_difference": {
        "display": "Demographic Parity Difference",
        "eu_ai_act": "Art. 10 (Data governance) + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2 / GOVERN 6.2",
        "iso_42001": "Annex A.6 (Fairness)",
        "description": "Difference in positive prediction rates across demographic groups.",
    },
    "equalized_odds_difference": {
        "display": "Equalized Odds Difference",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Max difference in TPR and FPR across groups.",
    },
    "equal_opportunity_difference": {
        "display": "Equal Opportunity Difference",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Difference in true positive rates across groups.",
    },
    "average_odds_difference": {
        "display": "Average Odds Difference",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Average of TPR and FPR differences across groups.",
    },
    "disparate_impact_ratio": {
        "display": "Disparate Impact Ratio",
        "eu_ai_act": "Art. 10 + Art. 5(1)(d)",
        "nist_ai_rmf": "MEASURE 2.2",
        "iso_42001": "Annex A.6",
        "description": "Ratio of positive prediction rates — the 4/5ths rule threshold.",
    },
    # Privacy metrics
    "membership_inference_attack_score": {
        "display": "Membership Inference Risk",
        "eu_ai_act": "Art. 9 (Risk management) + GDPR Art. 25",
        "nist_ai_rmf": "MEASURE 2.6 / MANAGE 2.4",
        "iso_42001": "Annex A.7 (Privacy)",
        "description": "Attacker advantage in inferring training set membership.",
    },
    # Explainability
    "shap_feature_importance": {
        "display": "SHAP Feature Importance",
        "eu_ai_act": "Art. 13 (Transparency)",
        "nist_ai_rmf": "GOVERN 1.7 / MEASURE 2.9",
        "iso_42001": "Clause 8.4 (Transparency)",
        "description": "SHAP values indicating each feature's contribution to predictions.",
    },
    # Data quality / drift
    "population_stability_index": {
        "display": "Population Stability Index",
        "eu_ai_act": "Art. 9 (Risk management) + Art. 17",
        "nist_ai_rmf": "MEASURE 2.7",
        "iso_42001": "Clause 9.1",
        "description": "Measures distributional shift between reference and current data.",
    },
    # Robustness
    "adversarial_attack_success_rate": {
        "display": "Adversarial Attack Success Rate",
        "eu_ai_act": "Art. 15 (Robustness & cybersecurity)",
        "nist_ai_rmf": "MEASURE 2.8",
        "iso_42001": "Annex A.8 (Security)",
        "description": "Fraction of adversarial inputs that fool the model.",
    },
}


def get_mapping(metric_key: str) -> dict:
    """Return the standards mapping for a given metric key, or an empty dict."""
    return STANDARDS_MAP.get(metric_key, {})


def get_all_mappings() -> dict:
    """Return the full standards map."""
    return STANDARDS_MAP
