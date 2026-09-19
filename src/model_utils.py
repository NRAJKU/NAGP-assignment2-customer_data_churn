from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "TotalServices", "IsNewCustomer", "AvgMonthlySpend"
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create deterministic, row-level features without using the target."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        present_service_cols = [c for c in SERVICE_COLUMNS if c in df.columns]
        if present_service_cols:
            df["TotalServices"] = (df[present_service_cols] == "Yes").sum(axis=1).astype(float)
        else:
            df["TotalServices"] = 0.0

        df["IsNewCustomer"] = (pd.to_numeric(df["tenure"], errors="coerce") <= 12).astype(float)

        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        total = pd.to_numeric(df["TotalCharges"], errors="coerce")
        safe_tenure = tenure.replace(0, np.nan)
        df["AvgMonthlySpend"] = (total / safe_tenure).replace([np.inf, -np.inf], np.nan)
        df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(monthly)
        return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)


def build_model_pipeline(tree_params: dict) -> Pipeline:
    return Pipeline([
        ("features", CustomerFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=42, **tree_params)),
    ])


def transformed_feature_names(fitted_pipeline: Pipeline) -> np.ndarray:
    return fitted_pipeline.named_steps["preprocess"].get_feature_names_out()


def validate_required_columns(df: pd.DataFrame, include_target: bool = True) -> list[str]:
    required = set(CATEGORICAL_FEATURES + ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"])
    if include_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required - set(df.columns))
    return missing
