"""Streamlit front end for the saved Cleveland classification experiment."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from heart.config import ARTIFACT_DIR, BOUNDS, CATEGORIES, FEATURES, LABELS, REPORT_DIR, ROOT
from heart.predict import load_artifact, predict_rows

st.set_page_config(page_title="Heart Disease Classification", page_icon="🫀", layout="wide")
st.title("Heart Disease Classification")
st.caption("Cleveland dataset · Model comparison and interactive demonstration")
st.info("Educational demonstration using historical data. Outputs are model classifications, not medical diagnoses or future heart-attack risk estimates.")


@st.cache_resource
def saved_model(fingerprint):
    return load_artifact()


try:
    stamp = (ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8")
    model, metadata = saved_model(stamp)
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(f"Model unavailable: {exc}")
    st.code("python -m heart.train")
    st.stop()

demo, evaluation, about = st.tabs(["Try the model", "Evaluation", "Data & method"])

with demo:
    st.subheader("Enter a sample record")
    st.write("The starting values are an artificial example. Some features require clinical tests. Leave a value unknown if unavailable; the pipeline will impute it from training data.")
    defaults = {"age": 54, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0,
                "slope": 2, "ca": 0, "thal": 3}
    with st.form("sample"):
        columns = st.columns(3)
        inputs = {}
        for index, name in enumerate(FEATURES):
            with columns[index % 3]:
                if name in CATEGORIES:
                    options = list(CATEGORIES[name]) + ["Unknown"]
                    value = st.selectbox(LABELS[name], options, index=options.index(defaults[name]),
                                         format_func=lambda v, n=name: "Unknown" if v == "Unknown" else CATEGORIES[n][v], key=name)
                    inputs[name] = np.nan if value == "Unknown" else value
                else:
                    low, high = BOUNDS[name]
                    bounds = metadata["training_ranges"][name]
                    inputs[name] = st.number_input(LABELS[name], min_value=float(low), max_value=float(high),
                                                   value=float(defaults[name]), step=0.1 if name == "oldpeak" else 1.0,
                                                   format="%.1f" if name == "oldpeak" else "%.0f", key=name,
                                                   help=f"Training data range: {bounds['min']:g} to {bounds['max']:g}. This is not a clinical reference range.")
                    if st.checkbox("Unknown", key=f"missing_{name}"):
                        inputs[name] = np.nan
        submitted = st.form_submit_button("Classify sample", type="primary")
    if submitted:
        try:
            result = predict_rows(model, pd.DataFrame([inputs]), metadata["threshold"]).iloc[0]
            outside = [LABELS[n] for n, r in metadata["training_ranges"].items()
                       if pd.notna(inputs[n]) and not r["min"] <= inputs[n] <= r["max"]]
            if outside:
                st.warning("Outside the training range: " + ", ".join(outside) + ". This prediction extrapolates beyond observed values.")
            missing_count = int(pd.isna(pd.Series(inputs)).sum())
            if missing_count:
                st.warning(f"{missing_count} missing input(s) were imputed. More missing information can make the output less informative.")
            st.subheader("Predicted dataset class: " + ("disease present" if result.predicted_class else "disease absent"))
            st.metric("Model score for class 1", f"{result.model_score:.3f}")
            st.caption(f"Threshold: {metadata['threshold']:.2f}. This uncalibrated score is not a validated personal probability.")
        except ValueError as exc:
            st.error(str(exc))
    st.download_button("Download example input CSV", (ROOT / "data/examples/demo_input.csv").read_bytes(),
                       file_name="demo_input.csv", mime="text/csv")

with evaluation:
    st.subheader("Results from the saved experiment")
    st.write(f"Selected model: **{metadata['model'].replace('_', ' ')}**. "
             f"Training: **{metadata['train_rows']}** rows. Test: **{metadata['test_rows']}** rows.")
    metrics = metadata["test_metrics"]
    cols = st.columns(4)
    for column, label, key in zip(cols, ["Test ROC-AUC", "Precision", "Recall", "F1"], ["roc_auc", "precision", "recall", "f1"]):
        column.metric(label, f"{metrics[key]:.3f}")
    lo, hi = metrics["roc_auc_bootstrap_95_interval"]
    st.caption(f"Test ROC-AUC bootstrap interval: {lo:.3f}–{hi:.3f}; conditional on this model and small test sample. Precision, recall and F1 refer to disease-present class.")
    st.write("Models were selected using training cross-validation only. The test set was reserved for the selected model.")
    table = pd.DataFrame(metadata["cv_comparison"])[["model", "roc_auc_mean", "roc_auc_std"]]
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Fold standard deviation is variability across five validation folds, not a confidence interval.")
    for filename in ["evaluation.png", "importance.png"]:
        path = REPORT_DIR / "figures" / filename
        if path.exists():
            st.image(str(path), use_container_width=True)
    st.caption("Permutation importance describes this fitted model. It does not identify causes of disease; correlated features can share importance.")
    st.download_button("Download evaluation metadata", json.dumps(metadata, indent=2),
                       file_name="evaluation_metadata.json", mime="application/json")

with about:
    st.subheader("What the experiment measures")
    st.write("Binary classification of the original UCI target: 0 means absence; 1–4 map to presence. The Cleveland file contains 303 records and 13 input features.")
    st.markdown("[Dataset and citation](https://archive.ics.uci.edu/dataset/45/heart+disease) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)")
    st.write("Numeric values use training-median imputation and scaling. Categories use training-mode imputation and one-hot encoding. All preprocessing is fitted inside each training fold.")
    st.write("The candidates are a dummy baseline, logistic regression, random forest and XGBoost (unless explicitly skipped during training). The highest mean training CV ROC-AUC wins. The decision threshold is fixed at 0.5.")
    st.write("Limitations: small historical sample, no external validation, uncalibrated scores, and measurements that may require specialist tests. No treatment recommendations are produced.")
    st.write("Entered samples are processed in memory by the app; this code does not save them.")
    st.caption("Dataset: Janosi, Steinbrunn, Pfisterer & Detrano (1989), UCI Heart Disease, DOI 10.24432/C52P4X.")
