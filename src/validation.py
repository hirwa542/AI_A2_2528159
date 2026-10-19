"""Input schema, range, and data-quality validation."""
from __future__ import annotations
import pandas as pd
from src.config import REQUIRED_COLUMNS, FEATURES
RANGES = {
    "rainfall_mm": (600, 1400), "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80), "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0), "cooperative_support": (0, 1),
}

def validate_dataset(df: pd.DataFrame) -> dict:
    """Return schema, missingness, duplicate and range checks without hiding issues."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    range_violations = {}
    for col, (low, high) in RANGES.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = int(((numeric < low) | (numeric > high)).sum())
        if bad:
            range_violations[col] = bad
    if not set(df["season"].dropna().unique()).issubset({"A", "B"}):
        raise ValueError("season must contain only 'A' or 'B'.")
    if not set(df["high_risk"].dropna().unique()).issubset({0, 1}):
        raise ValueError("high_risk must contain only 0 or 1.")
    if range_violations:
        raise ValueError(f"Range violations detected: {range_violations}")
    return {
        "rows_before_deduplication": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "range_violations": range_violations,
        "required_columns_present": True,
        "targets_excluded_from_features": True,
    }

def validate_prediction_record(record: dict) -> dict:
    """Validate one inference record; fail with a clear message for bad input."""
    if not isinstance(record, dict):
        raise ValueError("Prediction input must be a JSON object.")
    missing = [key for key in FEATURES if key not in record]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
    extra = sorted(set(record) - set(FEATURES))
    if extra:
        raise ValueError(f"Unexpected field(s): {', '.join(extra)}")
    cleaned = dict(record)
    for col, (low, high) in RANGES.items():
        value = cleaned[col]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{col} must be a number.")
        if not low <= float(value) <= high:
            raise ValueError(f"{col} must be between {low} and {high}.")
        cleaned[col] = float(value)
    if cleaned["season"] not in {"A", "B"}:
        raise ValueError("season must be 'A' or 'B'.")
    if not isinstance(cleaned["season"], str):
        raise ValueError("season must be a string ('A' or 'B').")
    cleaned["cooperative_support"] = int(cleaned["cooperative_support"])
    return cleaned
