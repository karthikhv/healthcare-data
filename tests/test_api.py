import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "age": 58,
    "tumor_size": 28.4,
    "mean_radius": 14.5,
    "mean_texture": 20.4,
    "mean_smoothness": 0.098,
    "mean_compactness": 0.142,
}


@pytest.fixture(scope="module")
def payload():
    return SAMPLE_PAYLOAD


def test_predict_risk(payload):
    response = client.post("/predict-risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "malignant_probability" in body
    assert 0 <= body["malignant_probability"] <= 1


def test_predict_survival(payload):
    response = client.post("/predict-survival", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["median_survival_months"] > 0
    assert 0 <= body["survival_probability_36mo"] <= 1
    assert 0 <= body["survival_probability_60mo"] <= 1


def test_explain(payload):
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "top_contributions" in body
    assert len(body["top_contributions"]) > 0
