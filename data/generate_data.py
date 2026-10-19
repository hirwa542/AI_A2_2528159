"""Create Hirwa Honorine's deterministic synthetic potato-cooperative dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REG_DIGITS = "2528159"
DEFAULT_SEED = 8159
DATA_PATH = ROOT / "data" / f"AI_A2_{REG_DIGITS}.csv"


def generate_dataset(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate the assignment-specified rows, missing values, and duplicates."""
    if not 0 <= seed <= 9999:
        raise ValueError("Seed must be an integer from 0000 to 9999.")
    rng = np.random.default_rng(seed)
    rows = 240 + (seed % 41)
    data = pd.DataFrame({
        "rainfall_mm": rng.uniform(600, 1400, rows),
        "avg_temp_c": rng.uniform(12, 26, rows),
        "soil_moisture_pct": rng.uniform(20, 80, rows),
        "fertilizer_kg": rng.uniform(0, 250, rows),
        "field_size_ha": rng.uniform(0.2, 4.0, rows),
        "season": rng.choice(["A", "B"], rows),
        "cooperative_support": rng.integers(0, 2, rows),
    })
    season_effect = np.where(data["season"].to_numpy() == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, rows)
    yield_tons = (
        0.004 * data["rainfall_mm"].to_numpy()
        + 0.06 * data["soil_moisture_pct"].to_numpy()
        + 0.02 * data["fertilizer_kg"].to_numpy()
        + 1.30 * data["field_size_ha"].to_numpy()
        - 0.25 * np.abs(data["avg_temp_c"].to_numpy() - 18)
        + 0.70 * data["cooperative_support"].to_numpy()
        + season_effect + noise
    )
    data["yield_tons"] = np.clip(yield_tons, 0.50, None)
    yield_per_hectare = data["yield_tons"] / data["field_size_ha"]
    data["high_risk"] = (yield_per_hectare < np.median(yield_per_hectare)).astype(int)

    # Deliberately introduce the exact data-quality problems required by the brief.
    missing_count = round(0.05 * rows)
    soil_positions = rng.choice(rows, size=missing_count, replace=False)
    fertilizer_positions = rng.choice(rows, size=missing_count, replace=False)
    data.loc[soil_positions, "soil_moisture_pct"] = np.nan
    data.loc[fertilizer_positions, "fertilizer_kg"] = np.nan
    data = pd.concat([data, data.iloc[:3].copy()], ignore_index=True)
    return data


def write_dataset(seed: int = DEFAULT_SEED, output: Path = DATA_PATH) -> Path:
    """Write CSV using the required four-decimal numeric format."""
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(seed).to_csv(output, index=False, float_format="%.4f")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    path = write_dataset(args.seed, args.output)
    print(f"Created {path}")
    print(f"Rows including deliberate duplicates: {len(pd.read_csv(path))}")
