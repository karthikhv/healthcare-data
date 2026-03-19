from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[55])
    tumor_size: float = Field(..., gt=0, le=120, examples=[24.5])
    mean_radius: float = Field(..., gt=0, examples=[14.5])
    mean_texture: float = Field(..., gt=0, examples=[19.2])
    mean_smoothness: float = Field(..., gt=0, examples=[0.10])
    mean_compactness: float = Field(..., gt=0, examples=[0.14])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 57,
                "tumor_size": 26.1,
                "mean_radius": 14.6,
                "mean_texture": 20.1,
                "mean_smoothness": 0.095,
                "mean_compactness": 0.135,
            }]
        }
    }


class RiskResponse(BaseModel):
    predicted_label: str
    malignant_probability: float
    benign_probability: float
    metrics: Dict[str, float]
    model_version: str


class SurvivalResponse(BaseModel):
    median_survival_months: float
    survival_probability_36mo: float
    survival_probability_60mo: float
    model_version: str


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ExplainResponse(BaseModel):
    predicted_label: str
    malignant_probability: float
    top_contributions: List[FeatureContribution]
    model_version: str
