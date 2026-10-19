"""FastAPI inference service for the classroom decision-support prototype."""
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from src.config import FULL_NAME, REGISTRATION_NUMBER, STUDENT_ID, SEED, VERIFICATION_CODE, ROOT
from src.validation import validate_prediction_record
from predict import predict_record
import json

app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0", description="Synthetic-data classroom decision-support API. Not for consequential decisions.")

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(ge=0.2, le=4.0)
    season: Literal["A", "B"]
    cooperative_support: int = Field(ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "ok", "service": "Musanze Harvest Intelligence Service", "verification_code": VERIFICATION_CODE}

@app.get("/model-info")
def model_info():
    path = ROOT / "artifacts" / "model_info.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Models are not trained yet. Run run_all.py first.")
    return json.loads(path.read_text(encoding="utf-8"))

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_record(request.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
