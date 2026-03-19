# OncoPredict API

FastAPI service for cancer risk classification, survival analysis, and SHAP-based explanations using the UCI Breast Cancer dataset with SEER-like synthetic survival features.

## Folder structure
```
app/              # FastAPI app, schemas, pipeline utils
models/           # Serialized joblib artifacts (created by training)
data/             # Placeholder for data assets
tests/            # Pytest endpoint tests
train.py          # Training script to build models and explainer
requirements.txt  # Dependencies
Dockerfile        # Containerization for deployment
```

## Setup & local run
1) Create environment and install deps:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) Train and persist artifacts (writes to `models/`):
```bash
python train.py
```
3) Start the API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
4) Open docs at `http://localhost:8000/docs`.

## API samples
**Request body (all endpoints):**
```json
{
  "age": 57,
  "tumor_size": 26.1,
  "mean_radius": 14.6,
  "mean_texture": 20.1,
  "mean_smoothness": 0.095,
  "mean_compactness": 0.135
}
```

### POST /predict-risk
Response example:
```json
{
  "predicted_label": "malignant",
  "malignant_probability": 0.73,
  "benign_probability": 0.27,
  "metrics": {
    "accuracy": 0.96,
    "precision": 0.95,
    "recall": 0.97,
    "f1": 0.96,
    "roc_auc": 0.98
  },
  "model_version": "1.0.0"
}
```

### POST /predict-survival
Response example:
```json
{
  "median_survival_months": 84.1,
  "survival_probability_36mo": 0.89,
  "survival_probability_60mo": 0.77,
  "model_version": "1.0.0"
}
```

### POST /explain
Response example (top SHAP contributions):
```json
{
  "predicted_label": "malignant",
  "malignant_probability": 0.73,
  "top_contributions": [
    {"feature": "num__mean_radius", "contribution": 0.41},
    {"feature": "num__mean_texture", "contribution": 0.22},
    {"feature": "cat__tumor_size_bin_35+", "contribution": 0.19},
    {"feature": "num__tumor_size", "contribution": 0.14},
    {"feature": "cat__age_group_60-70", "contribution": 0.10}
  ],
  "model_version": "1.0.0"
}
```

## Testing
```bash
pytest
```

## Deploying on Render
- **Service type:** Web Service (Docker).
- **Repository root:** this project.
- **Build command:** `pip install -r requirements.txt && python train.py`
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- **Environment variables:** set `PORT` to `8000` (Render default). No secrets required.
- **Health check path:** `/predict-risk` (or `/docs`).
