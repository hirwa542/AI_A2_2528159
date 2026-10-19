"""Predict harvest yield, high-risk class, and feature-only cluster from JSON."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.config import ROOT, FEATURES, VERIFICATION_CODE, SEED
from src.validation import validate_prediction_record

def predict_record(record: dict) -> dict:
    cleaned = validate_prediction_record(record)
    X = pd.DataFrame([cleaned], columns=FEATURES)
    regression = joblib.load(ROOT / "models" / "regression_model.joblib")
    classification = joblib.load(ROOT / "models" / "classification_model.joblib")
    clustering = joblib.load(ROOT / "models" / "clustering_model.joblib")
    yield_estimate = float(regression.predict(X)[0])
    risk = int(classification.predict(X)[0])
    probability = float(classification.predict_proba(X)[0, 1])
    cluster_features = clustering["preprocessor"].transform(X)
    cluster = int(clustering["model"].predict(cluster_features)[0])
    return {"estimated_yield_tons": round(yield_estimate, 4), "high_risk_prediction": risk,
            "high_risk_probability": round(probability, 4), "cluster": cluster,
            "verification_code": VERIFICATION_CODE}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="Path to one JSON object with the seven feature fields")
    args = parser.parse_args()
    try:
        record = json.loads(Path(args.json).read_text(encoding="utf-8"))
        print(json.dumps(predict_record(record), indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Prediction input error: {exc}\n")
if __name__ == "__main__": main()
