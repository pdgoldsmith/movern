"""
Script to train and pickle all four Movern demo models using real public datasets.

Run once to generate the static .pkl and .csv files committed to the repo:

    python -m movern.datasets.demo_models.train_demos

Datasets used:
  credit     — Statlog German Credit Data   (UCI ID 144, CC BY 4.0, 1 000 rows)
  hiring     — Adult Census Income          (UCI ID 2,   CC BY 4.0, 48 842 rows)
  healthcare — Diabetes 130-US Hospitals    (UCI ID 296, CC BY 4.0, 101 766 rows)
  fraud      — Credit Card Fraud Detection  (OpenML 1597, ODbL,     284 807 rows)

Requires: ucimlrepo  (pip install ucimlrepo)
Output goes to: movern/datasets/demo_models/models/
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numerics and ordinal-encodes categoricals."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# 1. Credit Risk — Statlog German Credit Data (UCI ID 144)
# ---------------------------------------------------------------------------

def _train_credit():
    from ucimlrepo import fetch_ucirepo

    print("  Downloading German Credit Data from UCI…")
    ds = fetch_ucirepo(id=144)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze().copy()

    # ucimlrepo returns generic names Attribute1–Attribute20.
    # Rename to descriptive labels per the dataset documentation.
    X = X.rename(columns={
        "Attribute1":  "checking_status",
        "Attribute2":  "duration_months",
        "Attribute3":  "credit_history",
        "Attribute4":  "purpose",
        "Attribute5":  "credit_amount",
        "Attribute6":  "savings_status",
        "Attribute7":  "employment_since",
        "Attribute8":  "installment_rate_pct",
        "Attribute9":  "personal_status",
        "Attribute10": "other_debtors",
        "Attribute11": "residence_since",
        "Attribute12": "property",
        "Attribute13": "age",
        "Attribute14": "other_installment_plans",
        "Attribute15": "housing",
        "Attribute16": "existing_credits",
        "Attribute17": "job",
        "Attribute18": "num_dependents",
        "Attribute19": "telephone",
        "Attribute20": "foreign_worker",
    })

    # Target: 1 = good credit → 0, 2 = bad credit → 1
    y = (y == 2).astype(int)
    y.name = "default"

    # Sensitive features kept in original form for fairness analysis
    sensitive = X[["age", "personal_status"]].rename(
        columns={"personal_status": "sex_and_marital_status"}
    )

    preprocessor = _make_preprocessor(X)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)

    metadata = {
        "name": "Credit Risk",
        "description": (
            "Predicts whether a loan applicant is a good or bad credit risk "
            "based on financial history, employment, and demographics."
        ),
        "features": list(X.columns),
        "sensitive_features": list(sensitive.columns),
        "target": "default",
        "eu_ai_act_risk": "High Risk",
        "eu_ai_act_articles": ["Art. 9", "Art. 10", "Art. 13"],
        "dataset": {
            "name": "Statlog (German Credit Data)",
            "source": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            "license": "CC BY 4.0",
            "n_rows": 1000,
            "n_cols": 20,
            "description": (
                "1 000 loan applicants from a German bank, each labelled as a good or bad "
                "credit risk across 20 financial and demographic attributes. One of the most "
                "widely used datasets in algorithmic fairness research."
            ),
            "citation": (
                "Hofmann, H. (1994). Statlog (German Credit Data). "
                "UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77"
            ),
        },
        "assessment_data": {
            "description": (
                "Each row represents one loan applicant. Features cover checking account status, "
                "loan duration, credit history, loan purpose, savings, employment tenure, "
                "instalment rate, personal status, guarantors, property, age, housing, "
                "existing credits, job type, and telephone/foreign-worker indicators."
            ),
            "split": "30% stratified holdout (random seed 42)",
            "positive_label": "1 = bad credit risk (default)",
            "negative_label": "0 = good credit risk",
            "sensitive_feature_notes": (
                "Age is continuous (years). Personal/marital status encodes sex combined "
                "with marital status as categorical codes."
            ),
        },
    }
    return pipe, X_test, y_test, s_test, metadata


# ---------------------------------------------------------------------------
# 2. Hiring Screener — Adult Census Income (UCI ID 2)
# ---------------------------------------------------------------------------

def _train_hiring():
    from ucimlrepo import fetch_ucirepo

    print("  Downloading Adult Census Income from UCI…")
    ds = fetch_ucirepo(id=2)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze().copy()

    # Drop sampling weight — not a genuine feature
    X = X.drop(columns=["fnlwgt"], errors="ignore")

    # Strip whitespace that UCI Adult is known to contain
    for col in X.select_dtypes("object").columns:
        X[col] = X[col].str.strip()
    y = y.str.strip() if hasattr(y, "str") else y

    # Target: >50K → 1, ≤50K → 0
    y = y.str.strip(".").map(lambda v: 1 if ">50K" in str(v) else 0).astype(int)
    y.name = "income_over_50k"

    # Sensitive features
    sensitive_cols = ["sex", "race"]
    sensitive = X[sensitive_cols].copy()

    # Use a stratified 15 000-row sample for tractable training
    X_sample, _, y_sample, _, s_sample, _ = train_test_split(
        X, y, sensitive, train_size=15_000, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = _make_preprocessor(X_sample)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_sample, y_sample, s_sample,
        test_size=0.3, random_state=RANDOM_STATE, stratify=y_sample
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)

    metadata = {
        "name": "Hiring Screener",
        "description": (
            "Predicts whether an individual's annual income exceeds $50 000 based on "
            "US Census data. Used as a proxy for employment and promotion decisions."
        ),
        "features": list(X_sample.columns),
        "sensitive_features": sensitive_cols,
        "target": "income_over_50k",
        "eu_ai_act_risk": "High Risk",
        "eu_ai_act_articles": ["Art. 9", "Art. 10", "Art. 13"],
        "dataset": {
            "name": "Adult Census Income (UCI Adult)",
            "source": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/dataset/2/adult",
            "license": "CC BY 4.0",
            "n_rows": 48842,
            "n_cols": 14,
            "description": (
                "Extracted from the 1994 US Census by Barry Becker, this dataset contains "
                "48 842 individuals with demographic and employment attributes. The prediction "
                "task (income > $50K) is a standard benchmark for fairness across sex and race."
            ),
            "citation": (
                "Becker, B. & Kohavi, R. (1996). Adult. "
                "UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20"
            ),
        },
        "assessment_data": {
            "description": (
                "Each row represents one individual from the 1994 US Census. Features include "
                "age, education level, occupation, marital status, hours worked per week, "
                "capital gains/losses, and native country. The sampling weight column (fnlwgt) "
                "was excluded as it is not a predictive feature. "
                "The model's predictions are evaluated against the actual census income labels — "
                "accuracy and fairness metrics measure how well the model reproduces those "
                "real-world outcomes. "
                "Important caveat: the ground-truth label is reported census income, not an "
                "actual hiring decision. Fairness disparities reflect income inequality present "
                "in the 1994 census data, not the output of a real hiring process. Auditors "
                "should account for this when interpreting results."
            ),
            "split": "30% stratified holdout of a 15 000-row sample (random seed 42)",
            "positive_label": "1 = annual income > $50 000",
            "negative_label": "0 = annual income ≤ $50 000",
            "sensitive_feature_notes": (
                "Sex is binary (Male / Female). Race has five categories: White, Black, "
                "Asian-Pac-Islander, Amer-Indian-Eskimo, Other."
            ),
        },
    }
    return pipe, X_test, y_test, s_test, metadata


# ---------------------------------------------------------------------------
# 3. Healthcare Outcome — Diabetes 130-US Hospitals (UCI ID 296)
# ---------------------------------------------------------------------------

def _train_healthcare():
    from ucimlrepo import fetch_ucirepo

    print("  Downloading Diabetes 130-US Hospitals from UCI…")
    ds = fetch_ucirepo(id=296)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze().copy()

    # Replace '?' with NaN
    X = X.replace("?", np.nan)

    # Drop columns with very high missing rates or non-predictive IDs
    drop_cols = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # Target: readmitted within 30 days → 1, otherwise → 0
    y = y.map(lambda v: 1 if str(v).strip() == "<30" else 0).astype(int)
    y.name = "readmitted_30d"

    # Sensitive features
    sensitive_cols = [c for c in ["race", "gender", "age"] if c in X.columns]
    sensitive = X[sensitive_cols].copy()

    # Use a stratified 15 000-row sample
    X_sample, _, y_sample, _, s_sample, _ = train_test_split(
        X, y, sensitive, train_size=15_000, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = _make_preprocessor(X_sample)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_sample, y_sample, s_sample,
        test_size=0.3, random_state=RANDOM_STATE, stratify=y_sample
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)

    metadata = {
        "name": "Healthcare Outcome",
        "description": (
            "Predicts 30-day hospital readmission for diabetic patients based on clinical "
            "encounter data including diagnoses, medications, and procedures."
        ),
        "features": list(X_sample.columns),
        "sensitive_features": sensitive_cols,
        "target": "readmitted_30d",
        "eu_ai_act_risk": "High Risk",
        "eu_ai_act_articles": ["Art. 9", "Art. 10", "Art. 13", "Art. 15"],
        "dataset": {
            "name": "Diabetes 130-US Hospitals (1999–2008)",
            "source": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
            "license": "CC BY 4.0",
            "n_rows": 101766,
            "n_cols": 47,
            "description": (
                "Clinical records for 101 766 diabetic inpatient encounters at 130 US hospitals "
                "over 10 years. Includes diagnoses, lab results, medications, and demographic "
                "attributes. Widely used for healthcare equity research."
            ),
            "citation": (
                "Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission "
                "Rates. BioMed Research International. "
                "UCI Repository: https://doi.org/10.24432/C5230J"
            ),
        },
        "assessment_data": {
            "description": (
                "Each row represents one hospital encounter for a diabetic patient. Features "
                "include time in hospital, number of lab procedures, number of medications, "
                "primary/secondary diagnoses (ICD-9 codes), HbA1c result, glucose test result, "
                "and dosage changes for 23 diabetes medications. Columns with >40% missing "
                "values (weight, payer code, medical specialty) were excluded."
            ),
            "split": "30% stratified holdout of a 15 000-row sample (random seed 42)",
            "positive_label": "1 = readmitted within 30 days",
            "negative_label": "0 = not readmitted or readmitted after 30 days",
            "sensitive_feature_notes": (
                "Race has five categories: Caucasian, AfricanAmerican, Hispanic, Asian, Other. "
                "Gender is binary. Age is encoded as decade ranges (e.g. [50-60))."
            ),
        },
    }
    return pipe, X_test, y_test, s_test, metadata


# ---------------------------------------------------------------------------
# 4. Fraud Detection — synthetic data (no licence concerns)
#
# Uses realistic synthetic transaction data rather than a real dataset to
# avoid ODbL share-alike obligations that apply to the ULB credit card fraud
# dataset. The synthetic data is generated deterministically from a fixed seed
# so results are fully reproducible.
# ---------------------------------------------------------------------------

def _train_fraud():
    rng = np.random.default_rng(RANDOM_STATE + 3)
    n = 4000

    amount = rng.exponential(scale=200, size=n)
    hour_of_day = rng.integers(0, 24, size=n).astype(float)
    merchant_category = rng.choice([0, 1, 2, 3, 4], size=n).astype(float)
    distance_from_home = rng.exponential(scale=50, size=n)
    num_transactions_24h = rng.integers(1, 20, size=n).astype(float)
    is_foreign = rng.choice([0, 1], size=n, p=[0.85, 0.15]).astype(float)

    log_odds = (
        -4.0
        + 0.002 * amount
        + 0.002 * distance_from_home
        + 0.08 * num_transactions_24h
        + 0.8 * is_foreign
        + 0.3 * ((hour_of_day < 5) | (hour_of_day > 22))
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="is_fraud")

    X = pd.DataFrame({
        "amount": amount,
        "hour_of_day": hour_of_day,
        "merchant_category": merchant_category,
        "distance_from_home": distance_from_home,
        "num_transactions_24h": num_transactions_24h,
        "is_foreign": is_foreign,
    })
    sensitive = pd.DataFrame(index=X.index)  # no sensitive features

    preprocessor = _make_preprocessor(X)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)

    metadata = {
        "name": "Fraud Detection",
        "description": (
            "Classifies payment transactions as fraudulent or legitimate based on "
            "transaction amount, timing, location, and behaviour patterns."
        ),
        "features": list(X.columns),
        "sensitive_features": [],
        "target": "is_fraud",
        "eu_ai_act_risk": "Limited Risk",
        "eu_ai_act_articles": ["Art. 13", "Art. 15"],
        "dataset": {
            "name": "Synthetic Payment Transactions",
            "source": "Generated for Movern",
            "url": "https://github.com/movern-ai/movern",
            "license": "Apache 2.0",
            "n_rows": n,
            "n_cols": 6,
            "description": (
                "Fully synthetic transaction dataset generated with a fixed random seed "
                "for reproducibility. Fraud probability is modelled as a function of "
                "transaction amount, distance from home, transaction velocity, foreign "
                "origin, and time of day. Used in place of a real fraud dataset to avoid "
                "share-alike licence obligations."
            ),
            "citation": "Movern project (synthetic). No external citation required.",
        },
        "assessment_data": {
            "description": (
                "Each row represents one synthetic payment transaction. 'amount' is the "
                "transaction value. 'hour_of_day' is the hour the transaction occurred (0–23). "
                "'merchant_category' is an encoded category (0–4). 'distance_from_home' is km "
                "from the cardholder's registered address. 'num_transactions_24h' is the number "
                "of transactions by that card in the prior 24 hours. 'is_foreign' flags whether "
                "the transaction occurred outside the cardholder's home country."
            ),
            "split": "30% stratified holdout (random seed 42)",
            "positive_label": "1 = fraudulent transaction",
            "negative_label": "0 = legitimate transaction",
            "sensitive_feature_notes": (
                "No sensitive demographic features. Fairness evaluators are not applicable."
            ),
        },
    }
    return pipe, X_test, y_test, s_test, metadata


# ---------------------------------------------------------------------------
# Main: train all and save
# ---------------------------------------------------------------------------

DEMO_TRAINERS = {
    "credit": _train_credit,
    "hiring": _train_hiring,
    "healthcare": _train_healthcare,
    "fraud": _train_fraud,
}


def train_all():
    try:
        import ucimlrepo  # noqa: F401
    except ImportError:
        raise ImportError(
            "ucimlrepo is required to regenerate demo models. "
            "Install it with: pip install ucimlrepo"
        )

    for name, trainer in DEMO_TRAINERS.items():
        print(f"Training {name}…")
        pipeline, X_test, y_test, sensitive_test, metadata = trainer()

        with open(MODELS_DIR / f"{name}_model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        X_test.to_csv(MODELS_DIR / f"{name}_X_test.csv", index=False)
        pd.Series(y_test, name="target").to_csv(MODELS_DIR / f"{name}_y_test.csv", index=False)
        sensitive_test.to_csv(MODELS_DIR / f"{name}_sensitive.csv", index=False)

        with open(MODELS_DIR / f"{name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {MODELS_DIR}/{name}_*  "
              f"(test set: {len(X_test)} rows)")

    print("Done.")


if __name__ == "__main__":
    train_all()
