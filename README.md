# Heart Disease Classification

A reproducible comparison of logistic regression, random forest and XGBoost on the processed Cleveland subset of UCI Heart Disease, with a dummy baseline and a Streamlit demonstration.

The task is to classify recorded disease presence from 13 features. It does not predict when a heart attack will occur. This is an educational experiment, not a clinically validated application.

## Run locally

Use **Python 3.12**. The small public dataset, trained pipeline and evaluation report are included, so retraining is optional for the first launch.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run these commands inside the extracted project folder. If the Windows Python launcher is unavailable, use `python` in the first command after confirming `python --version` reports 3.12. Full Windows/macOS/Linux instructions and troubleshooting are in [START_HERE.md](START_HERE.md).

## Reproduce the experiment

The commands below assume the project's virtual environment is active. You can instead use its full Python executable path as above.

```bash
python -m heart.download
python -m heart.explore
python -m heart.train
python -m pip install -r requirements-dev.txt
python -m pytest
```

`heart.download` verifies the bundled raw file and only accesses UCI if the file is missing. No Kaggle account or API key is required. `heart.train --skip-xgboost` supports a reduced comparison; it replaces the saved artifact and reports with that run's results.

## Method

- A fixed, stratified 80/20 split reserves 61 of 303 rows for final testing.
- All four candidates use the same five training folds. No hyperparameter search is performed.
- Numeric inputs use median imputation and standardization; categorical inputs use mode imputation and one-hot encoding. Vessel count (`ca`) is treated as a numeric count.
- The complete preprocessing pipeline is fitted separately within each fold.
- Highest mean training CV ROC-AUC selects the model; exact ties preserve candidate order.
- The threshold is fixed at 0.5. Test data do not choose a model, parameters, features or threshold.
- The saved model is fitted on the training partition only, matching the evaluated model.

See the [generated evaluation report](reports/evaluation.md) for measured results, uncertainty and errors. `reports/test_predictions.csv` allows the metrics to be independently recomputed. Fold standard deviations are not confidence intervals. CV precision/recall/F1 are macro averages; final test metrics use the positive class.

![Evaluation](reports/figures/evaluation.png)

## Project files

```text
app.py                         Streamlit application
heart/config.py                Schema, fixed split and experiment settings
heart/data.py                  Source validation and train/test split
heart/download.py              Checksum-verified dataset recovery
heart/modeling.py              Preprocessing and candidate estimators
heart/train.py                 Model selection, evaluation and saved artifact
heart/explore.py               Training-only exploratory plots
heart/predict.py               Shared inference and CSV command line interface
data/raw/                     Unchanged processed Cleveland source file
data/provenance.json           Source, license and SHA-256 checksum
data/examples/demo_input.csv   Artificial sample using original UCI codes
artifacts/                    Saved full pipeline and matching metadata
reports/                      Metrics, row splits, predictions and figures
notebooks/01_exploration.ipynb Training-only exploratory notebook
tests/                        Data, pipeline, artifact and application checks
docs/                         Data dictionary, deployment and project walkthrough
.github/workflows/tests.yml   Reproduce training and run checks in GitHub Actions
```

## Batch inference

```bash
python -m heart.predict data/examples/demo_input.csv --output predictions.csv
```

Input CSVs must contain exactly the 13 named feature columns. Column order can vary. Empty values or `?` are imputed; an entirely missing record is rejected. Unknown category codes are rejected to avoid mixing Kaggle and original UCI encodings. See [the data dictionary](docs/DATA_DICTIONARY.md).

## Limitations

The data are small and historical, and no external dataset was used to establish generalization. Scores are uncalibrated model outputs. Some inputs require clinical testing, limiting their value for an everyday screening tool. Feature importance describes associations within a model and cannot establish medical causes. The bootstrap interval conditions on the fixed fitted model and test sample; it excludes uncertainty from training and population differences.

## References and data license

Dataset: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. [DOI: 10.24432/C52P4X](https://doi.org/10.24432/C52P4X). The raw file is included unchanged under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The binary target transformation is documented in the code and data dictionary.

- [UCI dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Scikit-learn: preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn: evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/)
- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
