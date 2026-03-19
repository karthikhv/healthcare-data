from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: List[str] = [
    "age",
    "tumor_size",
    "mean_radius",
    "mean_texture",
    "mean_smoothness",
    "mean_compactness",
]

CAT_FEATURES: List[str] = ["age_group", "tumor_size_bin"]

AGE_BINS = [0, 40, 50, 60, 70, 120]
AGE_LABELS = ["<40", "40-50", "50-60", "60-70", "70+"]
TUMOR_BINS = [0, 15, 25, 35, 1_000]
TUMOR_LABELS = ["<15", "15-25", "25-35", "35+"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer that adds age_group and tumor_size_bin columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=NUMERIC_FEATURES)
        data["age_group"] = pd.cut(
            data["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
        ).astype(str)
        data["tumor_size_bin"] = pd.cut(
            data["tumor_size"], bins=TUMOR_BINS, labels=TUMOR_LABELS, right=False
        ).astype(str)
        return data

    def get_feature_names_out(self, input_features=None):
        return NUMERIC_FEATURES + CAT_FEATURES


def feature_engineering_transformer() -> FeatureEngineer:
    return FeatureEngineer()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def preprocess_payload(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([payload], columns=[*NUMERIC_FEATURES])


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return list(preprocessor.get_feature_names_out())
