# Musanze Harvest Intelligence Service

**Student:** Hirwa Honorine  
**Registration number:** 25/28159  
**Institutional username:** ug2528159@ines.ac.rw  
**Verification code:** AI-A2-8159  
**Seed:** 8159  
**Course:** SWE 3513 — Artificial Intelligence, Individual Assignment 2

## Problem
This individual project implements a reproducible classroom decision-support prototype for a fictional potato cooperative in Musanze. It estimates synthetic harvest yield, predicts a synthetic high-risk class, and groups records by similar feature profiles. **The dataset is synthetic and the output is not suitable for real farming or consequential decisions.**

## Requirements
- Python 3.11 or 3.12 recommended
- Windows 10/11, macOS, or Linux
- Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux activate with `source .venv/bin/activate`.

## Reproduce the full pipeline
Run from this project root:

```powershell
python run_all.py --student-id "ug2528159@ines.ac.rw" --seed 8159
```

The command recreates `data/AI_A2_2528159.csv`, validates data quality, removes exact duplicates for training, fits models with training-only preprocessing, compares baselines and candidates, selects K using silhouette scores, saves artifacts/models, and creates the technical report.

## Predict one record
```powershell
python predict.py --json samples/valid.json
```

The CLI returns an estimated yield, high-risk class, high-risk probability, cluster label, and verification code. `samples/invalid.json` is intentionally missing a required field and should return a clear error.

## Run tests
```powershell
pytest -q
```

The test suite covers deterministic generation, missingness and duplicates, valid inference, missing field, wrong type, out-of-range value, extra field, API health, valid API prediction, and invalid API requests.

## Run the API
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```
- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/model-info`
- `POST http://127.0.0.1:8000/predict`
- Interactive docs: `http://127.0.0.1:8000/docs`

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

## Main output files
- `artifacts/data_fingerprint.txt` — SHA-256 of the exact generated CSV bytes
- `artifacts/data_report.json` — missingness, duplicate and schema checks
- `artifacts/regression_metrics.json` — baseline/candidate MAE, RMSE and R²
- `artifacts/classification_metrics.json` — precision, recall, F1, confusion matrix and ROC-AUC
- `artifacts/clustering_report.json` — silhouette scores for K=2..6 and profiles
- `artifacts/test_predictions.csv` — test-row predictions, errors, probabilities and clusters
- `artifacts/plots/` — residual, confusion-matrix and cluster figures
- `models/` — saved models and fitted preprocessing
- `docs/AI_A2_2528159_REPORT.pdf` — technical report

## Reproducibility and limitations
The dataset is generated with `numpy.random.default_rng(8159)`. Model splits and stochastic estimators use seed 8159. Numeric imputation/scaling and categorical imputation/encoding are in scikit-learn pipelines fitted on training rows only. `yield_tons` and `high_risk` are excluded from model inputs. The data are generated from a fixed formula and cannot establish real-world performance, fairness, or causality. A cooperative officer or agronomist must review outputs; the prototype must not determine access to finance, insurance, public services, or penalties.

## Repository and final commit
Repository URL: **Add your own GitHub repository URL after publishing this exact project.**  
Final commit hash: **Fill in after your final commit.**  
Tested operating system: this package must be tested on your own computer; record the actual OS/Python version after running the commands.

## Submission evidence checklist
Before submitting, run the pipeline and tests on your own computer, inspect the generated report and plots, complete and sign the declaration yourself, create the test log from your actual run, and record the required continuous unedited demo video (maximum two minutes). Do not claim the video or declaration is complete until you personally create them.
