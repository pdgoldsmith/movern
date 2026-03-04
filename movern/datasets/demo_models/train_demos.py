"""
Script to train and pickle all four Movern demo models.

Run once to generate the static .pkl and .csv files committed to the repo:

    python -m movern.datasets.demo_models.train_demos

Output goes to: movern/datasets/demo_models/models/
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Credit Risk  (uses UCI default-of-credit-card data features)
# ---------------------------------------------------------------------------

def _make_credit_data():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 2000
    age = rng.integers(22, 70, size=n).astype(float)
    sex = rng.choice([0, 1], size=n, p=[0.45, 0.55]).astype(float)  # 0=female,1=male
    limit_bal = rng.uniform(10_000, 800_000, size=n)
    pay1 = rng.choice([-1, 0, 1, 2, 3], size=n).astype(float)
    bill1 = rng.uniform(0, 200_000, size=n)
    pay_amt1 = rng.uniform(0, 50_000, size=n)
    education = rng.choice([1, 2, 3], size=n, p=[0.3, 0.5, 0.2]).astype(float)

    # Outcome: older + lower-limit + late payment → higher default risk
    log_odds = (
        -1.5
        - 0.01 * (age - 40)
        - 0.000002 * limit_bal
        + 0.5 * pay1
        + 0.3 * (sex == 1)       # slight bias toward male defaults
        - 0.2 * education
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < prob).astype(int)

    X = pd.DataFrame({
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "AGE": age,
        "PAY_1": pay1,
        "BILL_AMT1": bill1,
        "PAY_AMT1": pay_amt1,
    })
    sensitive = pd.DataFrame({"SEX": sex, "AGE": age})
    return X, y, sensitive


def _train_credit():
    X, y, sensitive = _make_credit_data()
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    metadata = {
        "name": "Credit Risk",
        "description": "Predicts credit card default from payment history and demographics.",
        "features": list(X.columns),
        "sensitive_features": ["SEX", "AGE"],
        "target": "default_payment_next_month",
        "eu_ai_act_risk": "High Risk",
        "eu_ai_act_articles": ["Art. 9", "Art. 10", "Art. 13"],
    }
    return pipe, X_test, y_test, s_test, metadata


# ---------------------------------------------------------------------------
# 2. Hiring Screener
# ---------------------------------------------------------------------------

def _make_hiring_data():
    rng = np.random.default_rng(RANDOM_STATE + 1)
    n = 2000
    gender = rng.choice([0, 1], size=n, p=[0.48, 0.52]).astype(float)  # 0=female,1=male
    race = rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15]).astype(float)  # 0=white,1=Black,2=Hispanic
    yoe = rng.integers(0, 20, size=n).astype(float)
    degree = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3]).astype(float)
    gap_years = rng.integers(0, 5, size=n).astype(float)
    skills_score = rng.uniform(0, 100, size=n)

    log_odds = (
        -0.5
        + 0.1 * yoe
        + 0.3 * degree
        - 0.1 * gap_years
        + 0.02 * skills_score
        + 0.4 * (gender == 1)     # gender bias in favour of males
        - 0.6 * (race == 1)       # race bias against Black applicants
        - 0.4 * (race == 2)       # race bias against Hispanic applicants
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < prob).astype(int)

    X = pd.DataFrame({
        "years_of_experience": yoe,
        "highest_degree": degree,
        "employment_gap_years": gap_years,
        "skills_score": skills_score,
        "gender": gender,
        "race": race,
    })
    sensitive = pd.DataFrame({"gender": gender, "race": race})
    return X, y, sensitive


def _train_hiring():
    X, y, sensitive = _make_hiring_data()
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    metadata = {
        "name": "Hiring Screener",
        "description": "Screens job applicants for interview invitation.",
        "features": list(X.columns),
        "sensitive_features": ["gender", "race"],
        "target": "hired",
        "eu_ai_act_risk": "High Risk",
        "eu_ai_act_articles": ["Art. 9", "Art. 10", "Art. 13"],
    }
    return pipe, X_test, y_test, s_test, metadata


# ---------------------------------------------------------------------------
# 3. Healthcare Outcome
# ---------------------------------------------------------------------------

def _make_healthcare_data():
    rng = np.random.default_rng(RANDOM_STATE + 2)
    n = 2000
    age = rng.integers(18, 90, size=n).astype(float)
    sex = rng.choice([0, 1], size=n, p=[0.5, 0.5]).astype(float)
    race = rng.choice([0, 1, 2, 3], size=n, p=[0.55, 0.20, 0.15, 0.10]).astype(float)
    bmi = rng.uniform(18, 45, size=n)
    systolic_bp = rng.uniform(90, 180, size=n)
    num_medications = rng.integers(0, 10, size=n).astype(float)
    comorbidities = rng.integers(0, 5, size=n).astype(float)

    log_odds = (
        -3.0
        + 0.04 * (age - 50)
        + 0.02 * (bmi - 25)
        + 0.02 * (systolic_bp - 120)
        + 0.15 * num_medications
        + 0.3 * comorbidities
        + 0.2 * (race == 1)       # mild disparity
        + 0.1 * (age > 65)
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < prob).astype(int)

    X = pd.DataFrame({
        "age": age,
        "sex": sex,
        "race": race,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "num_medications": num_medications,
        "comorbidities": comorbidities,
    })
    sensitive = pd.DataFrame({"age": age, "sex": sex, "race": race})
    return X, y, sensitive


def _train_healthcare():
    X, y, sensitive = _make_healthcare_data()
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    metadata = {
        "name": "Healthcare Outcome",
        "description": "Predicts adverse health outcome for hospital readmission risk.",
        "features": list(X.columns),
        "sensitive_features": ["age", "sex", "race"],
        "target": "adverse_outcome",
        "eu_ai_act_risk": "High Risk",
        "eu_ai_act_articles": ["Art. 9", "Art. 10", "Art. 13", "Art. 15"],
    }
    return pipe, X_test, y_test, s_test, metadata


# ---------------------------------------------------------------------------
# 4. Fraud Detection
# ---------------------------------------------------------------------------

def _make_fraud_data():
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
    y = (rng.uniform(size=n) < prob).astype(int)

    X = pd.DataFrame({
        "amount": amount,
        "hour_of_day": hour_of_day,
        "merchant_category": merchant_category,
        "distance_from_home": distance_from_home,
        "num_transactions_24h": num_transactions_24h,
        "is_foreign": is_foreign,
    })
    sensitive = pd.DataFrame({"is_foreign": is_foreign})
    return X, y, sensitive


def _train_fraud():
    X, y, sensitive = _make_fraud_data()
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    metadata = {
        "name": "Fraud Detection",
        "description": "Classifies payment transactions as fraudulent or legitimate.",
        "features": list(X.columns),
        "sensitive_features": [],
        "target": "is_fraud",
        "eu_ai_act_risk": "Limited Risk",
        "eu_ai_act_articles": ["Art. 13", "Art. 15"],
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
    for name, trainer in DEMO_TRAINERS.items():
        print(f"Training {name}...")
        pipeline, X_test, y_test, sensitive_test, metadata = trainer()

        with open(MODELS_DIR / f"{name}_model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        X_test.to_csv(MODELS_DIR / f"{name}_X_test.csv", index=False)
        pd.Series(y_test, name="target").to_csv(MODELS_DIR / f"{name}_y_test.csv", index=False)
        sensitive_test.to_csv(MODELS_DIR / f"{name}_sensitive.csv", index=False)

        import json
        with open(MODELS_DIR / f"{name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {MODELS_DIR}/{name}_*")

    print("Done.")


if __name__ == "__main__":
    train_all()
