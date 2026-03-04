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
    if not model_path.exists():
        raise FileNotFoundError(
            f"Demo model files not found at {MODELS_DIR}. "
            "Run: python -m movern.datasets.demo_models.train_demos"
        )

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    X_test = pd.read_csv(MODELS_DIR / f"{name}_X_test.csv")
    y_test = pd.read_csv(MODELS_DIR / f"{name}_y_test.csv").squeeze("columns")
    sensitive = pd.read_csv(MODELS_DIR / f"{name}_sensitive.csv")

    with open(MODELS_DIR / f"{name}_metadata.json") as f:
        metadata = json.load(f)

    return pipeline, X_test, y_test, sensitive, metadata
