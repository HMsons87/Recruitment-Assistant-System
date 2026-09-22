from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.preprocessing import add_engineered_features
from src.modeling import build_models, calculate_metrics
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def make_synthetic_dataset(n=400, seed=42):
    rng = np.random.default_rng(seed)
    education = np.array(["Primary School", "High School", "Graduate", "Masters", "Phd"])
    company_size = np.array(["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
    company_type = np.array(["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Early Stage Startup", "Other"])
    university = np.array(["no_enrollment", "Full time course", "Part time course"])
    major = np.array(["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"])
    relevant = np.array(["Has relevent experience", "No relevent experience"])
    last_job = np.array(["never", "1", "2", "3", "4", ">4"])
    exp = np.array(["<1"] + [str(i) for i in range(1, 21)] + [">20"])

    df = pd.DataFrame({
        "enrollee_id": np.arange(100000, 100000+n),
        "city": rng.integers(1, 125, n),
        "city_development_index": rng.uniform(0.45, 0.95, n),
        "gender": rng.choice(["Male", "Female"], n),
        "relevent_experience": rng.choice(relevant, n),
        "enrolled_university": rng.choice(university, n),
        "education_level": rng.choice(education, n),
        "major_discipline": rng.choice(major, n),
        "experience": rng.choice(exp, n),
        "company_size": rng.choice(company_size, n),
        "company_type": rng.choice(company_type, n),
        "last_new_job": rng.choice(last_job, n),
        "training_hours": rng.integers(1, 200, n),
    })

    # Inject missing values.
    for col in ["education_level", "major_discipline", "company_size", "company_type", "experience"]:
        idx = rng.choice(n, size=max(1, n // 20), replace=False)
        df.loc[idx, col] = np.nan

    # Generate a non-trivial binary target.
    raw_score = (
        1.5 * (df["relevent_experience"] == "No relevent experience").astype(float)
        + 0.8 * (df["training_hours"] > 100).astype(float)
        - 1.0 * (df["city_development_index"] > 0.75).astype(float)
        + rng.normal(0, 1, n)
    )
    threshold = np.quantile(raw_score, 0.65)
    df["target"] = (raw_score > threshold).astype(int)
    return df


def main():
    df = make_synthetic_dataset()
    X = add_engineered_features(df.drop(columns=["target"]))
    X = X.drop(columns=["enrollee_id", "city", "gender"], errors="ignore")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    lr, rf = build_models()
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, pred, prob)
        assert all(np.isfinite(list(metrics.values())))
        assert 0 <= metrics["Accuracy"] <= 1
        assert 0 <= metrics["F1 Score"] <= 1
        assert 0 <= metrics["ROC-AUC"] <= 1
        print(name, metrics)

    # Verify inference with the same schema used by the Streamlit app.
    one_candidate = df.drop(columns=["target"]).iloc[[0]].copy()
    one_features = add_engineered_features(one_candidate)
    p = rf.predict_proba(one_features)[0, 1]
    assert 0 <= p <= 1

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
