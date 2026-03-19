import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from lifelines import CoxPHFitter
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.pipeline import (
    CAT_FEATURES,
    NUMERIC_FEATURES,
    FeatureEngineer,
    build_preprocessor,
    feature_engineering_transformer,
)

RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
METRICS_PATH = MODEL_DIR / "metrics.json"
RISK_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
SURVIVAL_MODEL_PATH = MODEL_DIR / "survival_model.joblib"
SHAP_EXPLAINER_PATH = MODEL_DIR / "shap_explainer.joblib"


def _synthesize_features() -> Tuple[pd.DataFrame, np.ndarray]:
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target = dataset.target

    rename_map = {
        "mean radius": "mean_radius",
        "mean texture": "mean_texture",
        "mean smoothness": "mean_smoothness",
        "mean compactness": "mean_compactness",
        "mean area": "mean_area",
    }
    df = df.rename(columns=rename_map)

    rng = np.random.default_rng(RANDOM_SEED)
    # Create synthetic clinical features to resemble SEER-like inputs
    radius_norm = (df["mean_radius"] - df["mean_radius"].min()) / (
        df["mean_radius"].max() - df["mean_radius"].min()
    )
    df["age"] = (35 + 30 * radius_norm + rng.normal(0, 4, size=len(df))).clip(25, 90)
    df["tumor_size"] = (np.sqrt(df["mean_area"]) + rng.normal(0, 3, size=len(df))).clip(5, 80)

    selected = df[
        ["age", "tumor_size", "mean_radius", "mean_texture", "mean_smoothness", "mean_compactness"]
    ]
    return selected, target


def _evaluation_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def _fit_risk_model(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray
) -> Tuple[Pipeline, Dict[str, float]]:
    preprocessor = build_preprocessor()
    features = feature_engineering_transformer()

    model = LogisticRegression(max_iter=400, class_weight="balanced")
    risk_pipeline = Pipeline(
        steps=[("feature_engineering", features), ("preprocess", preprocessor), ("model", model)]
    )
    risk_pipeline.fit(X_train, y_train)

    proba = risk_pipeline.predict_proba(X_test)[:, 0]  # class 0 = malignant
    preds = risk_pipeline.predict(X_test)
    metrics = _evaluation_metrics(y_test, preds, proba)
    return risk_pipeline, metrics


def _fit_survival_model(
    features_df: pd.DataFrame, target: np.ndarray
) -> CoxPHFitter:
    """Fit Cox PH model using only numeric features to avoid collinearity."""
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Use raw numeric features only (no one-hot encoding to avoid singularity)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[NUMERIC_FEATURES])
    X_df = pd.DataFrame(X_scaled, columns=NUMERIC_FEATURES)

    # Synthetic survival times: shorter durations for malignant cases
    baseline = 120 - (features_df["tumor_size"].values * 1.5)
    malignant_penalty = np.where(target == 0, 25, -5)  # dataset: 0 malignant, 1 benign
    durations = (baseline + malignant_penalty + rng.normal(0, 8, size=len(baseline))).clip(6, 180)
    events = rng.binomial(1, p=np.where(target == 0, 0.8, 0.4))

    surv_df = X_df.copy()
    surv_df["duration"] = durations
    surv_df["event"] = events

    cox = CoxPHFitter(penalizer=0.1)  # Add regularization to handle near-collinearity
    cox.fit(surv_df, duration_col="duration", event_col="event", show_progress=False)
    return cox


def _fit_shap_explainer(
    risk_pipeline: Pipeline, X_background: pd.DataFrame
) -> shap.Explainer:
    engineered = risk_pipeline.named_steps["feature_engineering"].transform(X_background)
    transformed = risk_pipeline.named_steps["preprocess"].transform(engineered)
    model = risk_pipeline.named_steps["model"]
    masker = shap.maskers.Independent(transformed)
    explainer = shap.Explainer(model, masker)
    return explainer


def train_models() -> Dict[str, float]:
    X, y = _synthesize_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    risk_pipeline, metrics = _fit_risk_model(X_train, y_train, X_test, y_test)
    cox_model = _fit_survival_model(X, y)
    background = X_train.sample(min(80, len(X_train)), random_state=RANDOM_SEED)
    shap_explainer = _fit_shap_explainer(risk_pipeline, background)

    joblib.dump(risk_pipeline, RISK_MODEL_PATH)
    joblib.dump(cox_model, SURVIVAL_MODEL_PATH)
    joblib.dump(shap_explainer, SHAP_EXPLAINER_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    train_models()
