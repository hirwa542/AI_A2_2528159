# Musanze Harvest Intelligence Service

**Student:** Hirwa Honorine  
**Registration number:** 25/28159  
**Institutional username:** ug2528159@ines.ac.rw  
**Verification code:** AI-A2-8159  
**Seed:** 8159  
**Course:** SWE 3513 — Artificial Intelligence, Individual Assignment 2

## Problem

This individual project implements a reproducible classroom decision-support prototype for a fictional potato cooperative in Musanze. It estimates synthetic harvest yield, predicts a synthetic high-risk class, and groups records by similar feature profiles.

**The dataset is synthetic, and the output is not suitable for real farming or consequential decisions.**

## Requirements

- Python 3.11 or 3.12 recommended
- Windows 10/11, macOS, or Linux
- Install dependencies in a virtual environment.

### Setup on Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate the virtual environment using:

```bash
source .venv/bin/activate
```

## Reproduce the Full Pipeline

Run from the project root directory:

```bash
python run_all.py --student-id "ug2528159@ines.ac.rw" --seed 8159
```

The command recreates `data/AI_A2_2528159.csv`, validates data quality, removes exact duplicates for training, fits models with training-only preprocessing, compares baselines and candidate models, selects the number of clusters using silhouette scores, saves artifacts and models, and creates the technical report.

## Predict One Record

```bash
python predict.py --json samples/valid.json
```

The CLI returns an estimated yield, high-risk class, high-risk probability, cluster label, and verification code.

The file `samples/invalid.json` is intentionally missing a required field and should return a clear error.

## Run Tests

```bash
python -m pytest -q
```

The test suite covers deterministic generation, missing values and duplicates, valid inference, missing fields, incorrect types, out-of-range values, extra fields, API health, valid API predictions, and invalid API requests.

## Run the API

Start the API with:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Available endpoints:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/model-info`
- `POST http://127.0.0.1:8000/predict`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Example JSON for `POST /predict`:

```json
{
  "rainfall_mm": 950,
  "avg_temp_c": 18,
  "soil_moisture_pct": 48,
  "fertilizer_kg": 110,
  "field_size_ha": 1.5,
  "season": "A",
  "cooperative_support": 1
}
```

## Main Output Files

- `artifacts/data_fingerprint.txt` — SHA-256 fingerprint of the generated CSV file
- `artifacts/data_report.json` — missing-value, duplicate, and schema checks
- `artifacts/regression_metrics.json` — baseline and candidate MAE, RMSE, and R²
- `artifacts/classification_metrics.json` — precision, recall, F1-score, confusion matrix, and ROC-AUC
- `artifacts/clustering_report.json` — silhouette scores for K=2 to K=6 and cluster profiles
- `artifacts/test_predictions.csv` — test-row predictions, errors, probabilities, and cluster labels
- `artifacts/plots/` — residual, confusion-matrix, and cluster plots
- `models/` — saved models and fitted preprocessing
- `docs/AI_A2_2528159_REPORT.pdf` — technical report

## Reproducibility and Limitations

The dataset is generated using `numpy.random.default_rng(8159)`. Model splits and stochastic estimators use seed 8159. Numeric imputation and scaling, together with categorical imputation and encoding, are implemented in scikit-learn pipelines fitted on training rows only.

The target columns `yield_tons` and `high_risk` are excluded from model inputs.

The data are generated from a fixed formula and cannot establish real-world performance, fairness, or causality. A cooperative officer or agronomist must review any outputs. The prototype must not determine access to finance, insurance, public services, or penalties.

## Repository and Final Commit

**Repository URL:** https://github.com/hirwa542/AI_A2_2528159.git

**Final commit hash:** ea0bef8
**Tested operating system and Python version:** Python 3.13.14


