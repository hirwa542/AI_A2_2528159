"""Reproducible data preparation, model training, evaluation, and reporting."""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import *
from src.validation import validate_dataset
from data.generate_data import write_dataset

ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
PLOTS = ARTIFACTS / "plots"


def _json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)], remainder="drop")


def _metric_regression(y_true, pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, pred)), "rmse": float(math.sqrt(mean_squared_error(y_true, pred))), "r2": float(r2_score(y_true, pred))}


def _metric_classification(y_true, pred, probability) -> dict:
    try:
        auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        auc = None
    return {"precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
            "roc_auc": auc}


def _plot_regression(y_true, pred) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    residuals = np.asarray(y_true) - np.asarray(pred)
    ax.scatter(pred, residuals, alpha=.75)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(title="Regression residuals on held-out test data", xlabel="Predicted yield (tons)", ylabel="Actual − predicted (tons)")
    fig.tight_layout(); fig.savefig(PLOTS / "residuals.png", dpi=150); plt.close(fig)


def _plot_confusion(cm) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(title="High-risk classification confusion matrix", xlabel="Predicted class", ylabel="Actual class", xticks=[0,1], yticks=[0,1])
    for (i,j), val in np.ndenumerate(np.asarray(cm)):
        ax.text(j, i, str(val), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=.046); fig.tight_layout(); fig.savefig(PLOTS / "confusion_matrix.png", dpi=150); plt.close(fig)


def _plot_clusters(features, labels, centers) -> None:
    # PCA is used only to make a two-dimensional illustration of feature profiles.
    from sklearn.decomposition import PCA
    scaled = StandardScaler().fit_transform(features)
    projection = PCA(n_components=2, random_state=SEED).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scatter = ax.scatter(projection[:,0], projection[:,1], c=labels, cmap="viridis", alpha=.8)
    ax.set(title="Farm-condition clusters (PCA view)", xlabel="PCA component 1", ylabel="PCA component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster label"); fig.tight_layout(); fig.savefig(PLOTS / "clusters.png", dpi=150); plt.close(fig)


def run_pipeline(student_id: str, seed: int) -> dict:
    """Run all assignment stages and regenerate deterministic outputs."""
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("Student ID must be your institutional username/email.")
    if seed != SEED:
        raise ValueError(f"For registration number {REGISTRATION_NUMBER}, --seed must be {SEED} (last four digits).")
    # Do not allow placeholder values to be recorded as student identity.
    if "YOUR_" in student_id.upper() or student_id.strip().lower() in {"student", "username", "your_actual_username"}:
        raise ValueError("Replace the student-id placeholder with your actual INES institutional username/email.")
    for folder in (ARTIFACTS, MODELS, PLOTS): folder.mkdir(parents=True, exist_ok=True)
    data_path = write_dataset(seed, DATA_PATH)
    fingerprint = _sha256(data_path)
    (ARTIFACTS / "data_fingerprint.txt").write_text(f"Student ID: {student_id.strip()}\nSeed: {seed}\nVerification code: {VERIFICATION_CODE}\nSHA-256: {fingerprint}\n", encoding="utf-8")
    raw = pd.read_csv(data_path)
    data_quality = validate_dataset(raw)
    clean = raw.drop_duplicates().reset_index(drop=True)
    quality = {**data_quality, "rows_after_deduplication": int(len(clean)), "duplicates_removed": int(len(raw)-len(clean)), "rows_with_any_missing_value": int(clean.isna().any(axis=1).sum()), "fingerprint_sha256": fingerprint, "student_id": student_id.strip(), "seed": seed, "verification_code": VERIFICATION_CODE}
    _json_write(ARTIFACTS / "data_report.json", quality)

    X = clean[FEATURES].copy()
    y_reg = clean[REGRESSION_TARGET].astype(float)
    y_cls = clean[CLASSIFICATION_TARGET].astype(int)
    all_indices = np.arange(len(clean))
    train_idx, test_idx = train_test_split(all_indices, test_size=0.25, random_state=seed, stratify=y_cls)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yr_train, yr_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    yc_train, yc_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]

    # Regression: mean baseline plus two candidates, with preprocessing fit only on training data.
    reg_models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=seed, n_jobs=1),
    }
    reg_results, fitted_reg = {}, {}
    for name, estimator in reg_models.items():
        model = Pipeline([("preprocess", _preprocessor()), ("model", estimator)])
        model.fit(X_train, yr_train)
        pred = model.predict(X_test)
        reg_results[name] = _metric_regression(yr_test, pred)
        fitted_reg[name] = model
    selected_reg = min(["linear_regression", "random_forest_regressor"], key=lambda n: reg_results[n]["rmse"])
    reg_results["selected_model"] = selected_reg
    reg_results["selection_reason"] = "Selected the candidate with lower held-out test RMSE; the mean baseline is retained for comparison."
    _json_write(ARTIFACTS / "regression_metrics.json", reg_results)
    reg_pred = fitted_reg[selected_reg].predict(X_test)
    _plot_regression(yr_test, reg_pred)
    joblib.dump(fitted_reg[selected_reg], MODELS / "regression_model.joblib")

    # Classification: majority baseline plus two candidates; target columns never enter X.
    cls_models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest_classifier": RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1),
    }
    cls_results, fitted_cls = {}, {}
    for name, estimator in cls_models.items():
        model = Pipeline([("preprocess", _preprocessor()), ("model", estimator)])
        model.fit(X_train, yc_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model[-1], "predict_proba") else pred.astype(float)
        cls_results[name] = _metric_classification(yc_test, pred, proba)
        fitted_cls[name] = model
    selected_cls = max(["logistic_regression", "random_forest_classifier"], key=lambda n: (cls_results[n]["f1"], cls_results[n]["roc_auc"] or -1))
    cls_results["selected_model"] = selected_cls
    cls_results["selection_reason"] = "Selected the candidate with the best held-out F1; ROC-AUC breaks ties when available."
    _json_write(ARTIFACTS / "classification_metrics.json", cls_results)
    cls_pred = fitted_cls[selected_cls].predict(X_test)
    cls_prob = fitted_cls[selected_cls].predict_proba(X_test)[:, 1]
    _plot_confusion(cls_results[selected_cls]["confusion_matrix"])
    joblib.dump(fitted_cls[selected_cls], MODELS / "classification_model.joblib")

    # Clustering: feature-only, training-fold preprocessing; evaluate K=2..6 on training fold.
    cluster_pre = _preprocessor()
    train_matrix = cluster_pre.fit_transform(X_train)
    test_matrix = cluster_pre.transform(X_test)
    cluster_scores, candidates = {}, {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(train_matrix)
        score = float(silhouette_score(train_matrix, labels))
        cluster_scores[str(k)] = {"silhouette_score": score, "inertia": float(km.inertia_)}
        candidates[k] = km
    selected_k = max(range(2,7), key=lambda k: cluster_scores[str(k)]["silhouette_score"])
    final_km = candidates[selected_k]
    train_labels = final_km.labels_
    all_matrix = cluster_pre.transform(X)
    all_labels = final_km.predict(all_matrix)
    clean_with_clusters = clean.copy(); clean_with_clusters["cluster"] = all_labels
    profiles=[]
    for cluster_id in sorted(np.unique(all_labels)):
        subset = clean_with_clusters.loc[clean_with_clusters["cluster"] == cluster_id]
        means = {col: round(float(subset[col].mean()), 3) for col in ["rainfall_mm","avg_temp_c","soil_moisture_pct","fertilizer_kg","field_size_ha","cooperative_support"]}
        season_counts = subset["season"].value_counts().to_dict()
        profiles.append({"cluster": int(cluster_id), "farms": int(len(subset)), "feature_means": means, "season_counts": {str(k):int(v) for k,v in season_counts.items()}, "plain_language_profile": _cluster_profile(means, len(subset))})
    cluster_report = {"scores_by_k": cluster_scores, "selected_k": int(selected_k), "selection_reason": "Selected K with the highest silhouette score measured on the training fold.", "profiles": profiles, "features_used": FEATURES, "targets_excluded": True}
    _json_write(ARTIFACTS / "clustering_report.json", cluster_report)
    _plot_clusters(train_matrix, train_labels, final_km.cluster_centers_)
    joblib.dump({"preprocessor": cluster_pre, "model": final_km, "selected_k": selected_k}, MODELS / "clustering_model.joblib")

    predictions = pd.DataFrame({
        "row_id": test_idx.tolist(), "actual_yield_tons": yr_test.to_numpy(), "predicted_yield_tons": reg_pred,
        "regression_error_tons": yr_test.to_numpy()-reg_pred, "actual_high_risk": yc_test.to_numpy(),
        "predicted_high_risk": cls_pred, "high_risk_probability": cls_prob,
        "cluster": final_km.predict(test_matrix),
    })
    predictions.to_csv(ARTIFACTS / "test_predictions.csv", index=False, float_format="%.4f")
    model_info = {"full_name": FULL_NAME, "registration_number": REGISTRATION_NUMBER, "student_id": student_id.strip(), "seed": seed, "verification_code": VERIFICATION_CODE, "dataset_sha256": fingerprint, "regression_model": selected_reg, "classification_model": selected_cls, "selected_cluster_count": int(selected_k), "dataset_rows_after_deduplication": int(len(clean)), "features": FEATURES}
    _json_write(ARTIFACTS / "model_info.json", model_info)
    _create_report(model_info, quality, reg_results, cls_results, cluster_report, predictions)
    return model_info


def _cluster_profile(means: dict, count: int) -> str:
    size = "larger" if count >= 80 else "medium-sized" if count >= 35 else "smaller"
    moisture = "higher" if means["soil_moisture_pct"] >= 50 else "lower"
    fertilizer = "higher" if means["fertilizer_kg"] >= 125 else "lower"
    area = "larger" if means["field_size_ha"] >= 2.1 else "smaller"
    return f"A {size} group of records with {moisture} average soil moisture, {fertilizer} fertilizer use, and {area} field size. These are descriptive patterns in synthetic data, not causal findings."


def _create_report(info, quality, reg, cls, clusters, predictions) -> None:
    """Create a seven-page technical PDF from the exact generated artifacts."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
    path = ROOT / "docs" / f"AI_A2_{REG_DIGITS}_REPORT.pdf"; path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.7*cm, leftMargin=1.7*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet(); styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17365D"), leading=24)); styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10)); styles.add(ParagraphStyle(name="SubHead", parent=styles["Heading2"], textColor=colors.HexColor("#17365D")))
    story=[]
    def p(text, style="BodyText"): story.append(Paragraph(text, styles[style])); story.append(Spacer(1, 7))
    def table(rows, widths=None):
        t=Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT"); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17365D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#B8C4D1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F2F5F9")]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])); story.append(t); story.append(Spacer(1, 9))
    p("MUSANZE HARVEST INTELLIGENCE SERVICE", "Cover")
    p("SWE 3513 · Artificial Intelligence · Individual Assignment 2", "Heading2")
    p(f"<b>Student:</b> {FULL_NAME}<br/><b>Registration number:</b> {REGISTRATION_NUMBER}<br/><b>Institutional username:</b> {info['student_id']}<br/><b>Verification code:</b> {VERIFICATION_CODE}<br/><b>Seed:</b> {SEED}<br/><b>Dataset SHA-256:</b> {info['dataset_sha256']}")
    p("Decision problem", "Heading2")
    p("The fictional Musanze potato cooperative needs a local decision-support prototype to estimate harvest quantity, flag records with relatively high production risk, and group farms with similar input conditions. This project demonstrates a reproducible machine-learning workflow and a small prediction API. All records are synthetic.")
    p("Questions addressed", "Heading3")
    p("1. What harvest quantity is estimated from the farm conditions? 2. Does a record fall into the synthetic high-risk class? 3. Which farms have similar feature profiles? These outputs support review; they are not guarantees about real harvests.")
    story.append(PageBreak())
    p("2. Data and reproducibility", "Heading1")
    p(f"The dataset uses numpy.random.default_rng({SEED}) and the lecturer's specified feature ranges and yield formula. The base row count is 240 + ({SEED} mod 41) = {240 + SEED%41}. The CSV includes the required deliberately introduced missing values and three appended duplicate rows. The SHA-256 fingerprint is computed from the exact final CSV bytes.")
    table([["Data check","Result"],["Rows before deduplication",str(quality['rows_before_deduplication'])],["Exact duplicate rows",str(quality['duplicate_rows'])],["Rows after duplicate removal",str(quality['rows_after_deduplication'])],["Rows with missing data",str(quality['rows_with_any_missing_value'])],["CSV SHA-256",info['dataset_sha256']]])
    p("Data dictionary", "Heading2")
    table([["Field","Role / rule"],["rainfall_mm","Numeric feature, uniform 600–1400"],["avg_temp_c","Numeric feature, uniform 12–26"],["soil_moisture_pct","Numeric feature, uniform 20–80; missing values deliberately added"],["fertilizer_kg","Numeric feature, uniform 0–250; missing values deliberately added"],["field_size_ha","Numeric feature, uniform 0.2–4.0"],["season","Categorical feature: A or B"],["cooperative_support","Binary feature: 0 or 1"],["yield_tons","Regression target, formula-based and clipped at 0.50"],["high_risk","Binary target: yield per hectare below median"]])
    p("Data limitations", "Heading2")
    p("The data are simulated from a known formula and do not represent observed farms. Performance on this dataset cannot establish real-world accuracy or fairness. Missing values are imputed during model fitting; exact duplicates are detected and removed before modeling.")
    story.append(PageBreak())
    p("3. Pipeline design and leakage controls", "Heading1")
    p("The pipeline recreates the CSV, computes its fingerprint, validates required columns and feature ranges, reports missingness and duplicates, removes duplicate rows, and creates a fixed stratified train/test split using random_state=8159. The split is performed before model preprocessing is fitted.")
    p("Preprocessing", "Heading2")
    p("Numeric features use a median imputer followed by standard scaling. Season uses most-frequent imputation and one-hot encoding with unknown categories ignored. These transformations are inside scikit-learn Pipelines and are fitted only on training records. The target columns yield_tons and high_risk are never included as input features.")
    p("Models compared", "Heading2")
    table([["Task","Baseline","Candidates"],["Regression","Mean prediction","Linear regression; random forest regressor"],["Classification","Most frequent class","Logistic regression; random forest classifier"],["Clustering","Not applicable","K-means for K=2,3,4,5,6; silhouette comparison"]])
    p("Reproducibility controls", "Heading2")
    p("The generator, train/test split, random forest estimators and K-means use the registration-derived seed. The model selection rule is written to the metrics JSON. Re-running the pipeline regenerates the CSV and artifacts from source.")
    story.append(PageBreak())
    p("4. Regression and classification", "Heading1")
    p("Regression results", "Heading2")
    rr=[["Model","MAE","RMSE","R²"]]
    for n in ["mean_baseline","linear_regression","random_forest_regressor"]:
        rr.append([n, f"{reg[n]['mae']:.4f}",f"{reg[n]['rmse']:.4f}",f"{reg[n]['r2']:.4f}"])
    table(rr)
    p(f"Selected regression model: <b>{reg['selected_model']}</b>. {reg['selection_reason']}")
    p("Residual interpretation", "Heading2")
    residuals=predictions["regression_error_tons"].to_numpy(); worst=int(np.argmax(np.abs(residuals)))
    p(f"On the held-out test set, the largest absolute regression error in the exported prediction table is {abs(float(residuals[worst])):.3f} tons (test row index {int(predictions.iloc[worst]['row_id'])}). A residual is actual yield minus predicted yield. Its sign indicates under-prediction when positive and over-prediction when negative. This is one synthetic example, not a statement about a real farmer.")
    p("Classification results", "Heading2")
    cr=[["Model","Precision","Recall","F1","ROC-AUC"]]
    for n in ["majority_baseline","logistic_regression","random_forest_classifier"]:
        m=cls[n]; cr.append([n,f"{m['precision']:.3f}",f"{m['recall']:.3f}",f"{m['f1']:.3f}","N/A" if m['roc_auc'] is None else f"{m['roc_auc']:.3f}"])
    table(cr)
    p(f"Selected classifier: <b>{cls['selected_model']}</b>. {cls['selection_reason']}")
    confusion_mat=np.array(cls[cls['selected_model']]['confusion_matrix']); fp=int(confusion_mat[0,1]); fn=int(confusion_mat[1,0])
    p(f"Confusion-matrix review: the selected classifier produced {fp} false positive(s) and {fn} false negative(s) on the test set. A false negative means a synthetic high-risk record was missed; a false positive means a record was flagged despite its generated label being low risk. Both error types require human review.")
    plot_row = Table([[Image(str(PLOTS / "residuals.png"), width=7.0*cm, height=4.0*cm), Image(str(PLOTS / "confusion_matrix.png"), width=7.0*cm, height=4.0*cm)]], colWidths=[7.4*cm, 7.4*cm], hAlign="LEFT")
    story.append(plot_row); story.append(Spacer(1, 5))
    story.append(PageBreak())
    p("5. Clustering and farm profiles", "Heading1")
    table([["K","Silhouette","Inertia"]]+[[k, f"{v['silhouette_score']:.4f}",f"{v['inertia']:.2f}"] for k,v in clusters['scores_by_k'].items()])
    p(f"Selected K = {clusters['selected_k']}, based on the highest training-fold silhouette score among K=2 through K=6. Silhouette values measure how well points fit their own cluster relative to other clusters; they do not prove that a group is a real agricultural category.")
    for profile in clusters["profiles"]:
        p(f"<b>Cluster {profile['cluster']} ({profile['farms']} records):</b> {profile['plain_language_profile']} Average feature values: {profile['feature_means']}.")
    p("Operational interpretation", "Heading2")
    p("The cluster labels can help an officer inspect records with similar conditions, for example to plan follow-up questions or field visits. The labels are arbitrary identifiers, not rankings of farms. Cluster patterns are descriptive and must not be interpreted as causes or as evidence of a farmer's ability.")
    story.append(Image(str(PLOTS / "clusters.png"), width=13.5*cm, height=7.8*cm)); story.append(Spacer(1, 4))
    story.append(PageBreak())
    p("6. Prediction CLI and API", "Heading1")
    p("The saved regression, classification and clustering artifacts are loaded by predict.py and app.py. Prediction input requires the seven feature fields. The CLI prints a JSON response containing estimated yield, risk class, high-risk probability, cluster label and verification code.")
    p("Endpoints", "Heading2")
    table([["Method / path","Purpose"],["GET /health","Returns service health and verification code"],["GET /model-info","Returns student code, data fingerprint, selected models and K"],["POST /predict","Validates a feature record and returns predictions"]])
    p("Example request", "Heading2")
    p('{"rainfall_mm": 950, "avg_temp_c": 18, "soil_moisture_pct": 48, "fertilizer_kg": 110, "field_size_ha": 1.5, "season": "A", "cooperative_support": 1}', "Code")
    p("Invalid or incomplete requests receive a validation error rather than a fabricated prediction. Automated tests cover input validation, deterministic generation, inference, and API responses. Start the API with: <font name='Courier'>uvicorn app:app --host 127.0.0.1 --port 8000</font>; interactive documentation is at /docs.")
    p("Testing summary", "Heading2")
    p("Run <font name='Courier'>pytest -q</font> from the project root. The evidence folder contains a test-log template/generated result file only after tests are run in the target environment; users should rerun tests on their own computer before submission.")
    story.append(PageBreak())
    p("7. Responsible AI, limitations and reflection", "Heading1")
    p("Intended use", "Heading2")
    p("This is an educational prototype for demonstrating regression, classification, clustering and validated inference. It may be used to explore synthetic records and to illustrate how a local cooperative decision-support workflow could be structured.")
    p("Limitations and prohibited use", "Heading2")
    p("The dataset is synthetic and generated from a fixed formula, so metrics may be optimistic or structurally biased by that formula. It excludes real weather variation, soil measurements, pests, seed varieties, storage loss, measurement error and local socioeconomic conditions. Do not use these predictions to allocate loans, insurance, public services, or penalties, and do not replace field inspection or agronomist judgment with the model.")
    p("Human oversight", "Heading2")
    p("Any future real deployment would require consent and governance for real farm records, data-quality review, independent validation on representative seasons and locations, bias/error monitoring, security controls, clear explanation to users, and an agronomist or cooperative officer who can reject model output.")
    p("Author reflection to personalize", "Heading2")
    p("Before submission, the student should review the saved metrics and plots and be able to explain one regression error, one classification error and one cluster in their own words. Add a short personal reflection after reviewing those actual outputs; do not claim the synthetic results are field evidence.")
    p("Identity and reproducibility", "Heading2")
    p(f"<b>{FULL_NAME}</b> · {REGISTRATION_NUMBER} · {info['student_id']}<br/><b>{VERIFICATION_CODE}</b><br/>Dataset SHA-256: {info['dataset_sha256']}<br/>Report generated from this pipeline's metrics and artifacts.")
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica",8); canvas.setFillColor(colors.grey); canvas.drawString(1.7*cm, .8*cm, "SWE 3513 · AI Assignment 2 · Synthetic-data prototype"); canvas.drawRightString(A4[0]-1.7*cm, .8*cm, f"Page {doc.page}"); canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
