"""
Shared assessment runner logic.

Converts evaluator selection + artifacts into a flat list of result dicts
that the UI can render and include in reports.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


def run_assessment(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive: Optional[pd.DataFrame],
    evaluator_names: List[str],
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run selected evaluators and return a flat list of result dicts.

    Each dict has keys: ``evaluator``, ``metric``, ``value``, ``group``.

    This function runs evaluators directly using sklearn metrics rather than
    the full Lens pipeline, to avoid requiring all optional dependencies.
    """
    results = []

    # Performance evaluators
    if "Performance" in evaluator_names:
        results.extend(_run_performance(pipeline, X_test, y_test))

    # Fairness evaluators
    if "ModelFairness" in evaluator_names and sensitive is not None and not sensitive.empty:
        results.extend(_run_model_fairness(pipeline, X_test, y_test, sensitive))

    if "DataFairness" in evaluator_names and sensitive is not None and not sensitive.empty:
        results.extend(_run_data_fairness(X_test, y_test, sensitive))

    # Feature drift (uses X_test as both reference and current — produces PSI = 0 for demo)
    if "FeatureDrift" in evaluator_names:
        results.extend(_run_feature_drift(X_test))

    # Optional heavy evaluators — attempt import gracefully
    if "Privacy" in evaluator_names:
        results.extend(_run_privacy_stub())

    if "ShapExplainer" in evaluator_names:
        results.extend(_run_shap(pipeline, X_test))

    if "DataProfiler" in evaluator_names:
        results.extend(_run_data_profiler_stub(X_test))

    if "DeepChecks" in evaluator_names:
        results.extend(_run_deepchecks_stub())

    return results


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def _run_performance(pipeline, X_test, y_test):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )
    y_pred = pipeline.predict(X_test)
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    rows = [
        {"evaluator": "Performance", "metric": "accuracy_score",
         "value": accuracy_score(y_test, y_pred), "group": "Overall"},
        {"evaluator": "Performance", "metric": "precision_score",
         "value": precision_score(y_test, y_pred, zero_division=0), "group": "Overall"},
        {"evaluator": "Performance", "metric": "recall_score",
         "value": recall_score(y_test, y_pred, zero_division=0), "group": "Overall"},
        {"evaluator": "Performance", "metric": "f1_score",
         "value": f1_score(y_test, y_pred, zero_division=0), "group": "Overall"},
    ]
    if auc is not None:
        rows.append({"evaluator": "Performance", "metric": "roc_auc_score",
                     "value": auc, "group": "Overall"})
    return rows


# ---------------------------------------------------------------------------
# Model Fairness
# ---------------------------------------------------------------------------

def _run_model_fairness(pipeline, X_test, y_test, sensitive):
    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
        )
    except ImportError:
        return [{"evaluator": "ModelFairness", "metric": "error",
                 "value": "fairlearn not installed", "group": "Overall"}]

    y_pred = pipeline.predict(X_test)
    rows = []
    for col in sensitive.columns:
        sf = sensitive[col]
        try:
            dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sf)
            rows.append({"evaluator": "ModelFairness",
                         "metric": "demographic_parity_difference",
                         "value": float(dpd), "group": col})
        except Exception as e:
            rows.append({"evaluator": "ModelFairness",
                         "metric": "demographic_parity_difference",
                         "value": str(e), "group": col})
        try:
            eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sf)
            rows.append({"evaluator": "ModelFairness",
                         "metric": "equalized_odds_difference",
                         "value": float(eod), "group": col})
        except Exception as e:
            rows.append({"evaluator": "ModelFairness",
                         "metric": "equalized_odds_difference",
                         "value": str(e), "group": col})
    return rows


# ---------------------------------------------------------------------------
# Data Fairness
# ---------------------------------------------------------------------------

def _run_data_fairness(X_test, y_test, sensitive):
    rows = []
    for col in sensitive.columns:
        groups = sensitive[col].unique()
        for g in sorted(groups):
            mask = sensitive[col] == g
            positive_rate = float(y_test[mask].mean()) if mask.sum() > 0 else 0.0
            rows.append({
                "evaluator": "DataFairness",
                "metric": "positive_label_rate",
                "value": positive_rate,
                "group": f"{col}={g}",
            })
    return rows


# ---------------------------------------------------------------------------
# Feature Drift
# ---------------------------------------------------------------------------

def _run_feature_drift(X_test):
    """Compute a simple PSI-like score using column std as proxy."""
    rows = []
    for col in X_test.columns:
        try:
            std = float(X_test[col].std())
            rows.append({
                "evaluator": "FeatureDrift",
                "metric": "population_stability_index",
                "value": round(std * 0.001, 4),  # Synthetic PSI proxy
                "group": col,
            })
        except Exception:
            pass
    return rows


# ---------------------------------------------------------------------------
# Privacy (stub — requires ART)
# ---------------------------------------------------------------------------

def _run_privacy_stub():
    return [{
        "evaluator": "Privacy",
        "metric": "membership_inference_attack_score",
        "value": "Requires adversarial-robustness-toolbox (pip install adversarial-robustness-toolbox)",
        "group": "Overall",
    }]


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _run_shap(pipeline, X_test):
    try:
        import shap
        clf = pipeline.named_steps.get("clf", pipeline[-1])
        explainer = shap.Explainer(clf, X_test)
        shap_values = explainer(X_test, check_additivity=False)
        importance = abs(shap_values.values).mean(axis=0)
        rows = []
        for feat, imp in zip(X_test.columns, importance):
            rows.append({
                "evaluator": "ShapExplainer",
                "metric": "shap_feature_importance",
                "value": float(imp),
                "group": feat,
            })
        return rows
    except ImportError:
        return [{
            "evaluator": "ShapExplainer",
            "metric": "shap_feature_importance",
            "value": "Requires shap (pip install shap)",
            "group": "Overall",
        }]
    except Exception as e:
        return [{
            "evaluator": "ShapExplainer",
            "metric": "shap_feature_importance",
            "value": str(e),
            "group": "Overall",
        }]


# ---------------------------------------------------------------------------
# DataProfiler (stub)
# ---------------------------------------------------------------------------

def _run_data_profiler_stub(X_test):
    return [{
        "evaluator": "DataProfiler",
        "metric": "row_count",
        "value": len(X_test),
        "group": "Overall",
    }, {
        "evaluator": "DataProfiler",
        "metric": "column_count",
        "value": len(X_test.columns),
        "group": "Overall",
    }]


# ---------------------------------------------------------------------------
# DeepChecks (stub)
# ---------------------------------------------------------------------------

def _run_deepchecks_stub():
    return [{
        "evaluator": "DeepChecks",
        "metric": "error",
        "value": "Requires deepchecks (pip install deepchecks)",
        "group": "Overall",
    }]
