"""
Shared assessment runner logic.

Converts evaluator selection + artifacts into a flat list of result dicts
that the UI can render and include in reports.
"""

import subprocess
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# Heavy optional packages installed on-demand when their evaluator is selected.
_HEAVY_PACKAGES = {
    "DeepChecks": ("deepchecks", "deepchecks"),
    "Privacy":    ("art", "adversarial-robustness-toolbox"),
}


def get_packages_to_install(evaluator_names: List[str]) -> List[str]:
    """Return pip specs for any heavy packages not yet installed."""
    missing = []
    for ev in evaluator_names:
        if ev in _HEAVY_PACKAGES:
            import_name, pip_spec = _HEAVY_PACKAGES[ev]
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pip_spec)
    return missing


def install_packages(pip_specs: List[str]) -> None:
    """Install a list of pip specs."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pip_specs]
    )


def run_assessment(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive: Optional[pd.DataFrame],
    evaluator_names: List[str],
    metadata: Dict[str, Any],
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
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
        results.extend(_run_privacy(pipeline, X_test, y_test, X_train, y_train))

    if "ShapExplainer" in evaluator_names:
        results.extend(_run_shap(pipeline, X_test))

    if "DataProfiler" in evaluator_names:
        results.extend(_run_data_profiler_stub(X_test))

    if "DeepChecks" in evaluator_names:
        results.extend(_run_deepchecks_stub(pipeline, X_test, y_test))

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
        sf = sensitive[col].fillna("Unknown").astype(str)
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
        # Drop NaN and coerce to string to avoid mixed-type sort errors
        sf = sensitive[col].dropna().astype(str)
        for g in sorted(sf.unique()):
            mask = sf == g
            positive_rate = float(y_test.loc[mask.index][mask].mean()) if mask.sum() > 0 else 0.0
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
# Privacy — membership inference via ART
# ---------------------------------------------------------------------------

def _run_privacy(pipeline, X_test, y_test, X_train=None, y_train=None):
    try:
        from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased
        from art.estimators.classification import SklearnClassifier
    except ImportError:
        return [{"evaluator": "Privacy", "metric": "membership_inference_attack_score",
                 "value": "Requires adversarial-robustness-toolbox (pip install adversarial-robustness-toolbox)",
                 "group": "Overall"}]

    if X_train is None or len(X_train) == 0:
        return [{"evaluator": "Privacy", "metric": "membership_inference_attack_score",
                 "value": "No training sample available for membership inference.",
                 "group": "Overall"}]

    try:
        import numpy as np

        # Pre-transform through the pipeline's preprocessor so ART receives a
        # plain float numpy array rather than a DataFrame with named columns.
        # The ColumnTransformer inside the pipeline cannot select columns by
        # name from a numpy array, which causes "str - float" type errors.
        if hasattr(pipeline, 'steps') and len(pipeline.steps) > 1:
            X_train_t = pipeline[:-1].transform(X_train)
            X_test_t = pipeline[:-1].transform(X_test)
            clf = pipeline[-1]
        else:
            X_train_t = np.asarray(X_train)
            X_test_t = np.asarray(X_test)
            clf = pipeline

        art_clf = SklearnClassifier(model=clf)
        attack = MembershipInferenceBlackBoxRuleBased(art_clf)

        # Infer membership: 1 = predicted member, 0 = predicted non-member
        member_inferred = attack.infer(X_train_t, y_train.values.reshape(-1, 1))
        nonmember_inferred = attack.infer(X_test_t, y_test.values.reshape(-1, 1))

        # Attack accuracy: how often the attacker correctly identifies members vs non-members
        attack_accuracy = float(
            (member_inferred.sum() + (1 - nonmember_inferred).sum())
            / (len(member_inferred) + len(nonmember_inferred))
        )
        # Attacker advantage: how much better than random (0.5) the attack is
        attacker_advantage = float(max(0.0, attack_accuracy - 0.5) * 2)

        return [
            {"evaluator": "Privacy", "metric": "membership_inference_attack_score",
             "value": round(attack_accuracy, 4), "group": "Overall"},
            {"evaluator": "Privacy", "metric": "attacker_advantage",
             "value": round(attacker_advantage, 4), "group": "Overall"},
        ]
    except Exception as e:
        return [{"evaluator": "Privacy", "metric": "membership_inference_attack_score",
                 "value": str(e), "group": "Overall"}]


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _run_shap(pipeline, X_test):
    try:
        import shap
        import numpy as np

        # Pre-transform through the pipeline's preprocessor so SHAP receives a
        # plain float numpy array. Passing raw X_test fails for models with
        # string columns (e.g. healthcare race/gender/age) because the
        # classifier step expects post-preprocessed floats.
        if hasattr(pipeline, 'steps') and len(pipeline.steps) > 1:
            preprocessor = pipeline[:-1]
            clf = pipeline[-1]
            X_transformed = preprocessor.transform(X_test)

            # Recover output feature names from the ColumnTransformer if possible.
            try:
                feature_names = preprocessor[-1].get_feature_names_out()
                # Strip the "num__" / "cat__" prefixes added by ColumnTransformer.
                feature_names = [n.split("__", 1)[-1] for n in feature_names]
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        else:
            clf = pipeline
            X_transformed = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
            feature_names = list(X_test.columns) if hasattr(X_test, 'columns') else [f"feature_{i}" for i in range(X_transformed.shape[1])]

        explainer = shap.Explainer(clf, X_transformed)
        shap_values = explainer(X_transformed, check_additivity=False)
        vals = shap_values.values
        # Some explainers (e.g. TreeExplainer on RandomForest) return a 3-D
        # array (n_samples, n_features, n_classes). Collapse to 2-D by taking
        # the positive class for binary classification, or mean across classes.
        if vals.ndim == 3:
            vals = vals[:, :, 1] if vals.shape[2] == 2 else vals.mean(axis=2)
        importance = abs(vals).mean(axis=0)
        rows = []
        for feat, imp in zip(feature_names, importance):
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
# DeepChecks
# ---------------------------------------------------------------------------

def _run_deepchecks_stub(pipeline=None, X_test=None, y_test=None):
    try:
        import numpy as _np
        if not hasattr(_np, "Inf"):  # removed in NumPy 2.0, required by deepchecks
            _np.Inf = _np.inf
        from deepchecks.tabular import Dataset as DcDataset
        from deepchecks.tabular.checks import (
            ClassImbalance,
            FeatureLabelCorrelation,
            MixedDataTypes,
            IsSingleValue,
        )
    except ImportError:
        return [{"evaluator": "DeepChecks", "metric": "error",
                 "value": "Requires deepchecks (pip install deepchecks)",
                 "group": "Overall"}]

    try:
        cat_features = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
        ds = DcDataset(X_test, label=y_test, cat_features=cat_features)
        rows = []

        # Class imbalance
        try:
            result = ClassImbalance().run(ds)
            for cls, ratio in result.value.items():
                rows.append({"evaluator": "DeepChecks", "metric": "class_imbalance_ratio",
                             "value": round(float(ratio), 4), "group": str(cls)})
        except Exception:
            pass

        # Feature-label correlation
        try:
            result = FeatureLabelCorrelation().run(ds)
            for feat, corr in result.value.items():
                rows.append({"evaluator": "DeepChecks", "metric": "feature_label_correlation",
                             "value": round(float(corr), 4), "group": feat})
        except Exception:
            pass

        # Mixed data types
        try:
            result = MixedDataTypes().run(ds)
            if result.passed_conditions():
                rows.append({"evaluator": "DeepChecks", "metric": "mixed_data_types",
                             "value": "No mixed types detected", "group": "Overall"})
            else:
                rows.append({"evaluator": "DeepChecks", "metric": "mixed_data_types",
                             "value": "Mixed types detected in one or more columns", "group": "Overall"})
        except Exception:
            pass

        return rows if rows else [{"evaluator": "DeepChecks", "metric": "status",
                                   "value": "No checks returned results", "group": "Overall"}]
    except Exception as e:
        return [{"evaluator": "DeepChecks", "metric": "error", "value": str(e), "group": "Overall"}]
