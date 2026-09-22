from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "city_development_index",
    "training_hours",
    "experience_years",
    "training_hours_log",
    "training_intensity",
    "is_company_missing",
    "is_company_type_missing",
    "cdi_x_experience",
    "experience_job_ratio",
    "is_low_cdi",
]

ORDINAL_FEATURES = [
    "education_level",
    "company_size",
    "last_new_job",
]

NOMINAL_FEATURES = [
    "relevent_experience",
    "enrolled_university",
    "major_discipline",
    "company_type",
]

ORDINAL_CATEGORIES = [
    [
        "Primary School",
        "High School",
        "Graduate",
        "Masters",
        "Phd",
    ],
    [
        "<10",
        "10/49",
        "50-99",
        "100-500",
        "500-999",
        "1000-4999",
        "5000-9999",
        "10000+",
    ],
    ["never", "1", "2", "3", "4", ">4"],
]


def _numeric_pipeline(scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _ordinal_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def _nominal_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_pipeline(scale_numeric), NUMERIC_FEATURES),
            ("ord", _ordinal_pipeline(), ORDINAL_FEATURES),
            ("nom", _nominal_pipeline(), NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


from xgboost import XGBClassifier

def build_models(scale_pos_weight: float = 2.0) -> tuple[Pipeline, Pipeline, Pipeline]:
    # 1. Logistic Regression Pipeline
    lr_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 2. Tuned Random Forest Pipeline
    rf_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=9,                    
                    min_samples_leaf=8,             
                    max_features="sqrt",
                    class_weight={0: 1.0, 1: 2.2},
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 3. XGBoost Pipeline
    xgb_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBClassifier(
                    n_estimators=180,
                    max_depth=4,
                    learning_rate=0.06,
                    scale_pos_weight=1.9,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    return lr_pipeline, rf_pipeline, xgb_pipeline


def calculate_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
