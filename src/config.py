"""Shared configuration and schema for the individual assignment."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FULL_NAME = "Hirwa Honorine"
REGISTRATION_NUMBER = "25/28159"
STUDENT_ID = "ug2528159@ines.ac.rw"
SEED = 8159
VERIFICATION_CODE = "AI-A2-8159"
REG_DIGITS = "2528159"
DATA_PATH = ROOT / "data" / f"AI_A2_{REG_DIGITS}.csv"
FEATURES = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "season", "cooperative_support"]
NUMERIC_FEATURES = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "cooperative_support"]
CATEGORICAL_FEATURES = ["season"]
REGRESSION_TARGET = "yield_tons"
CLASSIFICATION_TARGET = "high_risk"
REQUIRED_COLUMNS = FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
