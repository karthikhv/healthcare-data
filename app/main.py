from pathlib import Path
from typing import List
import json

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from app.schemas import (
    ExplainResponse,
    FeatureContribution,
    PatientFeatures,
    RiskResponse,
    SurvivalResponse,
)
from app.pipeline import NUMERIC_FEATURES
from sklearn.preprocessing import StandardScaler
from train import METRICS_PATH, SHAP_EXPLAINER_PATH, SURVIVAL_MODEL_PATH, RISK_MODEL_PATH, train_models

app = FastAPI(title="OncoPredict API", version="1.0.0")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def ensure_artifacts():
    if not (RISK_MODEL_PATH.exists() and SURVIVAL_MODEL_PATH.exists() and SHAP_EXPLAINER_PATH.exists()):
        train_models()


ensure_artifacts()
risk_model = joblib.load(RISK_MODEL_PATH)
survival_model = joblib.load(SURVIVAL_MODEL_PATH)
shap_explainer: shap.Explainer = joblib.load(SHAP_EXPLAINER_PATH)
metrics = {} if not METRICS_PATH.exists() else json.loads(METRICS_PATH.read_text())

# Fit scaler for survival model input (matches training)
survival_scaler = StandardScaler()


def _transform_input(payload: PatientFeatures) -> pd.DataFrame:
    df = pd.DataFrame([payload.model_dump()])
    engineered = risk_model.named_steps["feature_engineering"].transform(df)
    transformed = risk_model.named_steps["preprocess"].transform(engineered)
    feature_names = risk_model.named_steps["preprocess"].get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    return df, transformed_df


def _survival_input(payload: PatientFeatures) -> pd.DataFrame:
    """Prepare input for survival model (numeric features only, scaled)."""
    df = pd.DataFrame([payload.model_dump()])[NUMERIC_FEATURES]
    # Use survival model's built-in means (approximate with zero-mean assumption)
    return df


def _malignant_probability(proba_vector: np.ndarray) -> float:
    # Class 0 corresponds to malignant in the UCI dataset
    return float(proba_vector[0])


@app.post("/predict-risk", response_model=RiskResponse, tags=["prediction"])
def predict_risk(payload: PatientFeatures):
    base_df = pd.DataFrame([payload.dict()])
    proba = risk_model.predict_proba(base_df)[0]
    malignant_prob = _malignant_probability(proba)
    benign_prob = float(proba[1])
    predicted_label = "malignant" if malignant_prob >= 0.5 else "benign"
    return RiskResponse(
        predicted_label=predicted_label,
        malignant_probability=malignant_prob,
        benign_probability=benign_prob,
        metrics=metrics,
        model_version="1.0.0",
    )


@app.post("/predict-survival", response_model=SurvivalResponse, tags=["prediction"])
def predict_survival(payload: PatientFeatures):
    surv_df = _survival_input(payload)
    try:
        median = float(survival_model.predict_median(surv_df).values[0])
        surv_probs = survival_model.predict_survival_function(surv_df, times=[36, 60])
        prob_36 = float(surv_probs.iloc[0].values[0])
        prob_60 = float(surv_probs.iloc[1].values[0])
    except Exception as exc:  # pragma: no cover - propagate as HTTP error
        raise HTTPException(status_code=500, detail=f"Survival prediction failed: {exc}") from exc
    return SurvivalResponse(
        median_survival_months=median,
        survival_probability_36mo=prob_36,
        survival_probability_60mo=prob_60,
        model_version="1.0.0",
    )


@app.post("/explain", response_model=ExplainResponse, tags=["explainability"])
def explain(payload: PatientFeatures):
    base_df, transformed_df = _transform_input(payload)
    proba = risk_model.predict_proba(base_df)[0]
    malignant_prob = _malignant_probability(proba)
    predicted_label = "malignant" if malignant_prob >= 0.5 else "benign"

    explanation = shap_explainer(transformed_df)
    feature_names = transformed_df.columns
    contributions = explanation.values[0]
    top_indices = np.argsort(np.abs(contributions))[::-1][:5]
    top_features: List[FeatureContribution] = [
        FeatureContribution(feature=feature_names[i], contribution=float(contributions[i]))
        for i in top_indices
    ]
    return ExplainResponse(
        predicted_label=predicted_label,
        malignant_probability=malignant_prob,
        top_contributions=top_features,
        model_version="1.0.0",
    )
