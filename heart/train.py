"""Compare predeclared pipelines on training folds, then evaluate one finalist."""
import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from heart.config import ARTIFACT_DIR, FEATURES, FOLDS, NUMERIC, RAW_PATH, REPORT_DIR, SEED, THRESHOLD
from heart.data import load_dataset, sha256, split_dataset
from heart.modeling import candidates


def auc_interval(y, scores, repetitions=2000):
    """Stratified percentile bootstrap; conditional on this fitted model/test sample."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    samples = []
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        samples.append(roc_auc_score(y[indices], scores[indices]))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def json_write(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-xgboost", action="store_true", help="Compare dummy, logistic regression and random forest only.")
    args = parser.parse_args()
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    for directory in (ARTIFACT_DIR, REPORT_DIR, REPORT_DIR / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    folds = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train))
    models = candidates(not args.skip_xgboost)
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy",
               "precision": make_scorer(precision_score, average="macro", zero_division=0),
               "recall": make_scorer(recall_score, average="macro", zero_division=0),
               "f1": make_scorer(f1_score, average="macro", zero_division=0)}
    # Macro metrics avoid an undefined positive-class precision for the dummy baseline.
    rows = []
    for name, pipeline in models.items():
        print(f"Cross-validating {name} ...", flush=True)
        values = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                n_jobs=1, error_score="raise")
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(values[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(values[f"test_{metric}"].std(ddof=1))
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORT_DIR / "cv_comparison.csv", index=False)
    # Stable ties preserve candidate order. The dummy remains eligible so failure to beat it is visible.
    winner = max(rows, key=lambda row: row["roc_auc_mean"])["model"]
    model = clone(models[winner]).fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    predicted = (scores >= THRESHOLD).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "roc_auc_bootstrap_95_interval": auc_interval(y_test, scores),
        "average_precision": float(average_precision_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "specificity": float(recall_score(y_test, predicted, pos_label=0)),
        "brier_score": float(brier_score_loss(y_test, scores)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "threshold": THRESHOLD,
        "test_rows": len(X_test),
    }
    json_write(REPORT_DIR / "test_metrics.json", metrics)
    pd.DataFrame({"source_row": X_test.index, "actual": y_test.to_numpy(), "score": scores,
                  "predicted": predicted}).to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    split = {"seed": SEED, "train_indices": X_train.index.tolist(), "test_indices": X_test.index.tolist(),
             "note": "Zero-based row positions in the unchanged raw file."}
    json_write(REPORT_DIR / "split.json", split)
    audit = {"rows": len(X), "features": len(FEATURES), "duplicates": int(X.duplicated().sum()),
             "missing_by_column": X.isna().sum().astype(int).to_dict(),
             "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
             "train_rows": len(X_train), "test_rows": len(X_test)}
    json_write(REPORT_DIR / "data_audit.json", audit)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["Absent", "Present"],
                                            ax=axes[0], colorbar=False, cmap="Greens")
    axes[0].set_title("Held-out test: counts")
    RocCurveDisplay.from_predictions(y_test, scores, ax=axes[1], name=winner.replace("_", " "))
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title("Held-out test: ROC curve")
    CalibrationDisplay.from_predictions(y_test, scores, n_bins=5, strategy="quantile", ax=axes[2])
    axes[2].set_title("Reliability (small test sample)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/evaluation.png", dpi=160)
    plt.close(fig)

    importance = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=20,
                                         random_state=SEED, n_jobs=1)
    importance_frame = pd.DataFrame({"feature": FEATURES, "auc_decrease_mean": importance.importances_mean,
                                     "auc_decrease_std": importance.importances_std}).sort_values("auc_decrease_mean")
    importance_frame.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_frame.feature, importance_frame.auc_decrease_mean,
            xerr=importance_frame.auc_decrease_std, color="#18786F")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Decrease in test ROC-AUC after shuffling", title="Permutation importance (descriptive only)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/importance.png", dpi=160)
    plt.close(fig)

    ranges = {name: {"min": float(X_train[name].min()), "max": float(X_train[name].max())} for name in NUMERIC}
    joblib.dump(model, ARTIFACT_DIR / "model.joblib", compress=3)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "streamlit", "matplotlib"]}
    if not args.skip_xgboost:
        versions["xgboost"] = importlib.metadata.version("xgboost")
    metadata = {
        "model": winner, "features": FEATURES, "threshold": THRESHOLD, "seed": SEED,
        "selection": "Highest mean training five-fold ROC-AUC; fixed candidates, no parameter search.",
        "cv_folds": FOLDS, "cv_comparison": rows, "train_rows": len(X_train), "test_rows": len(X_test),
        "training_ranges": ranges, "test_metrics": metrics, "raw_sha256": sha256(RAW_PATH),
        "model_sha256": sha256(ARTIFACT_DIR / "model.joblib"), "versions": versions,
        "python": platform.python_version(), "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "probabilities_calibrated": False,
        "limitations": "Small historical Cleveland sample; no external validation; educational use only.",
    }
    json_write(ARTIFACT_DIR / "metadata.json", metadata)
    # Artificial example, independent of the held-out sample; not an observed patient.
    example = pd.DataFrame([[54, 1, 2, 130, 240, 0, 0, 150, 0, 1.0, 2, 0, 3]], columns=FEATURES)
    (RAW_PATH.parent.parent / "examples").mkdir(exist_ok=True)
    example.to_csv(RAW_PATH.parent.parent / "examples/demo_input.csv", index=False)
    low, high = metrics["roc_auc_bootstrap_95_interval"]
    tn, fp, fn, tp = np.asarray(metrics["confusion_matrix"]).ravel()
    table = "\n".join(f"| {r['model']} | {r['roc_auc_mean']:.3f} | {r['roc_auc_std']:.3f} |" for r in rows)
    report = f"""# Evaluation report

Selected model: **{winner}**. Training rows: {len(X_train)}. Held-out test rows: {len(X_test)}.

## Model selection
Fixed candidates were compared on identical stratified five-fold splits of training data.
Preprocessing was fitted within each fold. Highest mean ROC-AUC determined the winner.
Cross-validation scores were used for selection and are not an unbiased final estimate.
No hyperparameter search or threshold optimization was performed. Threshold: {THRESHOLD}.

| Model | CV ROC-AUC mean | Fold standard deviation |
|---|---:|---:|
{table}

## Held-out evaluation
- ROC-AUC: {metrics['roc_auc']:.3f}; stratified bootstrap 95% interval: [{low:.3f}, {high:.3f}].
- Accuracy: {metrics['accuracy']:.3f}; positive-class precision: {metrics['precision']:.3f}.
- Positive-class recall: {metrics['recall']:.3f}; F1: {metrics['f1']:.3f}; specificity: {metrics['specificity']:.3f}.
- Average precision: {metrics['average_precision']:.3f}; Brier score: {metrics['brier_score']:.3f} (lower is better).
- True negatives: {tn}; false positives: {fp}; false negatives: {fn}; true positives: {tp}.

The interval reflects resampling this test sample with the model fixed; it excludes training,
model-selection and population-shift uncertainty. It is not a clinical validation interval.
CV precision/recall/F1 in the CSV are macro averages; test precision/recall/F1 above refer to class 1.

## Interpretation and limitations
Permutation importance is a post-evaluation description, not a feature-selection procedure.
Its error bars describe repeat-to-repeat shuffling variability, not confidence intervals.
Correlated features can share importance. Associations do not establish causes.
The reliability plot uses only five bins because the test set is small. Scores are uncalibrated.
This historical sample does not establish performance in modern or different populations.
The saved demo model is the evaluated training-only model; it was not refitted on the test set.
Repeatedly changing the project after inspecting test results would compromise the holdout.

![Evaluation](figures/evaluation.png)
![Permutation importance](figures/importance.png)
"""
    (REPORT_DIR / "evaluation.md").write_text(report, encoding="utf-8")
    print(f"Selected {winner}; held-out ROC-AUC={metrics['roc_auc']:.3f}. Reports and artifact saved.")


if __name__ == "__main__":
    main()
