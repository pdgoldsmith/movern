"""
Pre-trained demo models for Movern.

Usage::

    from movern.datasets.demo_models import load_demo_model

    model, assessment_data, metadata = load_demo_model("credit")
"""

import json
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"

AVAILABLE_MODELS = ["credit", "hiring", "healthcare", "fraud"]


def load_demo_model(name: str) -> Tuple[object, pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """Load a pre-trained Movern demo model.

    Parameters
    ----------
    name : str
        One of ``"credit"``, ``"hiring"``, ``"healthcare"``, ``"fraud"``.

    Returns
    -------
    pipeline : sklearn Pipeline
        Trained sklearn pipeline.
    X_test : pd.DataFrame
        Assessment feature matrix.
    y_test : pd.Series
        Assessment labels.
    sensitive : pd.DataFrame
        Sensitive feature columns for the assessment set.
    metadata : dict
        Model metadata (name, description, features, EU AI Act info, etc.).

    Examples
    --------
    >>> from movern.datasets.demo_models import load_demo_model
    >>> pipeline, X_test, y_test, sensitive, meta = load_demo_model("credit")
    >>> meta["name"]
    'Credit Risk'
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown demo model '{name}'. Available: {AVAILABLE_MODELS}"
        )

    model_path = MODELS_DIR / f"{name}_model.pkl"
    metadata_path = MODELS_DIR / f"{name}_metadata.json"

    # Auto-train if files are missing or metadata is from an older version
    # (detected by absence of the 'assessment_data' key).
    needs_train = not model_path.exists()
    if not needs_train and metadata_path.exists():
        import json as _json
        with open(metadata_path) as _f:
            _meta = _json.load(_f)
        if "assessment_data" not in _meta:
            needs_train = True

    if needs_train:
        from movern.datasets.demo_models.train_demos import DEMO_TRAINERS
        import pickle as _pickle
        import pandas as _pd
        import json as _json

        print(f"Building demo model '{name}' for the first time…")
        pipeline, X_test, y_test, sensitive_test, X_train_sample, y_train_sample, metadata = DEMO_TRAINERS[name]()

        with open(MODELS_DIR / f"{name}_model.pkl", "wb") as f:
            _pickle.dump(pipeline, f)
        X_test.to_csv(MODELS_DIR / f"{name}_X_test.csv", index=False)
        _pd.Series(y_test, name="target").to_csv(MODELS_DIR / f"{name}_y_test.csv", index=False)
        sensitive_test.to_csv(MODELS_DIR / f"{name}_sensitive.csv", index=False)
        X_train_sample.to_csv(MODELS_DIR / f"{name}_X_train_sample.csv", index=False)
        _pd.Series(y_train_sample, name="target").to_csv(MODELS_DIR / f"{name}_y_train_sample.csv", index=False)
        with open(metadata_path, "w") as f:
            _json.dump(metadata, f, indent=2)

        return pipeline, X_test, y_test, sensitive_test, metadata

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    X_test = pd.read_csv(MODELS_DIR / f"{name}_X_test.csv")
    y_test = pd.read_csv(MODELS_DIR / f"{name}_y_test.csv").squeeze("columns")
    sensitive_path = MODELS_DIR / f"{name}_sensitive.csv"
    try:
        sensitive = pd.read_csv(sensitive_path)
    except Exception:
        sensitive = pd.DataFrame()  # model has no sensitive features (e.g. fraud)

    try:
        X_train_sample = pd.read_csv(MODELS_DIR / f"{name}_X_train_sample.csv")
        y_train_sample = pd.read_csv(MODELS_DIR / f"{name}_y_train_sample.csv").squeeze("columns")
    except Exception:
        X_train_sample = pd.DataFrame()
        y_train_sample = pd.Series(dtype=int)

    with open(MODELS_DIR / f"{name}_metadata.json") as f:
        metadata = json.load(f)

    return pipeline, X_test, y_test, sensitive, X_train_sample, y_train_sample, metadata
