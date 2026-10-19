"""Functional and failure-path tests for the individual AI assignment."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from data.generate_data import generate_dataset, write_dataset
from src.config import ROOT, SEED, FEATURES, VERIFICATION_CODE
from src.validation import validate_prediction_record
from predict import predict_record
from app import app

client = TestClient(app)
VALID = {"rainfall_mm":950,"avg_temp_c":18,"soil_moisture_pct":48,"fertilizer_kg":110,"field_size_ha":1.5,"season":"A","cooperative_support":1}

def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_dataset(SEED), generate_dataset(SEED))

def test_generator_has_required_rows_missingness_and_duplicates():
    df=generate_dataset(SEED)
    assert len(df) == 240 + (SEED % 41) + 3
    assert df["soil_moisture_pct"].isna().sum() >= round(.05*(240+SEED%41))
    assert df["fertilizer_kg"].isna().sum() >= round(.05*(240+SEED%41))
    assert df.duplicated().sum() == 3

def test_valid_record_is_accepted():
    assert set(validate_prediction_record(VALID)) == set(FEATURES)

def test_missing_field_is_rejected():
    bad=dict(VALID); bad.pop("season")
    with pytest.raises(ValueError, match="Missing required"):
        validate_prediction_record(bad)

def test_wrong_numeric_type_is_rejected():
    bad=dict(VALID); bad["rainfall_mm"]="wet"
    with pytest.raises(ValueError, match="must be a number"):
        validate_prediction_record(bad)

def test_out_of_range_value_is_rejected():
    bad=dict(VALID); bad["rainfall_mm"]=1500
    with pytest.raises(ValueError, match="between 600 and 1400"):
        validate_prediction_record(bad)

def test_unexpected_field_is_rejected():
    bad=dict(VALID); bad["yield_tons"]=10
    with pytest.raises(ValueError, match="Unexpected field"):
        validate_prediction_record(bad)

def test_prediction_returns_required_fields():
    result=predict_record(VALID)
    assert set(result) == {"estimated_yield_tons","high_risk_prediction","high_risk_probability","cluster","verification_code"}
    assert np.isfinite(result["estimated_yield_tons"])
    assert 0 <= result["high_risk_probability"] <= 1
    assert result["verification_code"] == VERIFICATION_CODE

def test_health_api_response():
    response=client.get("/health")
    assert response.status_code == 200
    assert response.json()["verification_code"] == VERIFICATION_CODE

def test_api_valid_prediction():
    response=client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert "estimated_yield_tons" in response.json()

def test_api_invalid_range_returns_422():
    bad=dict(VALID); bad["rainfall_mm"]=2000
    response=client.post("/predict", json=bad)
    assert response.status_code == 422

def test_api_missing_field_returns_422():
    bad=dict(VALID); bad.pop("season")
    response=client.post("/predict", json=bad)
    assert response.status_code == 422
