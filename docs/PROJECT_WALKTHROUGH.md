# Understand and extend the project

## The data path

`data/raw` -> `heart.data` -> fixed train/test split -> fold-specific preprocessing -> candidate models -> CV selection -> one held-out evaluation -> saved full pipeline -> Streamlit / CSV inference.

Start by reading `config.py`, `data.py`, `modeling.py`, `train.py`, `predict.py`, then `app.py`.

## Questions you should be able to answer

**What does the target mean?** Recorded disease absence/presence in the historical dataset. It is not a five-year risk or future heart-attack outcome.

**Why reserve a test set?** Model selection should not benefit from the records used to report final performance. Training folds compare candidates; the held-out sample evaluates the chosen pipeline.

**Why use a Pipeline?** Each training fold learns its own imputation and scaling. Fitting these on the full dataset would let validation/test information influence training. Inference reuses the exact fitted transformations.

**Why include a dummy?** It establishes how a classifier performs without learning feature relationships. A sophisticated model must justify itself against a simple baseline.

**Why logistic regression?** It is a useful regularized baseline on small tabular datasets. A nonlinear ensemble is not automatically better.

**Why ROC-AUC?** It measures discrimination across score thresholds. It does not measure probability calibration. Precision, recall and the confusion matrix describe the fixed-threshold behavior.

**What is a false negative?** A disease-present record classified as absent. Open the test predictions and count these; discuss them alongside false positives. The app does not make treatment decisions.

**Why not optimize the threshold now?** This version fixes it at 0.5 before test evaluation. A threshold experiment must use training-side validation and a stated objective, not the test set.

**Why are some numeric codes one-hot encoded?** Codes such as chest-pain type are category identifiers; arithmetic distance between them has no assumed meaning. Vessel count is modeled numerically and is a possible sensitivity-analysis choice.

**Can a 0.8 score be called an 80% personal risk?** No. The model is uncalibrated and not externally validated, and the target is recorded disease presence.

**What do importance bars establish?** How shuffling an input changes this model's test ROC-AUC. They do not establish causality. The plot is descriptive after evaluation and must not be used to choose features against the same test set.

## A useful small extension

Compare logistic regression with `ca` treated as categorical versus numeric, using the exact same training folds. Preserve the original result. Record your hypothesis, code change, CV means and variability, and a brief interpretation in `docs/EXPERIMENT_NOTES.md`.

Do not claim a fresh unbiased test result after repeatedly inspecting and optimizing against this test set. For serious later model development, plan nested cross-validation or a genuinely new evaluation dataset. For this extension, a clearly labeled training-CV comparison is enough.

## 90-second demonstration

1. Explain the target and dataset limitation in one sentence.
2. Open Evaluation and explain which model won on training CV.
3. Read the held-out confusion matrix, including false negatives.
4. Classify the artificial default record, then mark a field unknown to show imputation.
5. Show `modeling.py` and explain how the same pipeline is used for training and inference.
6. Finish with the limitation or extension you investigated.

## Resume wording

After running, understanding and adapting the project, use a concise description such as:

> Developed and deployed a Streamlit heart disease classification demo; compared logistic regression, random forest and XGBoost using stratified cross-validation and evaluated the selected preprocessing pipeline on held-out Cleveland records.

Add a measured metric only if you can explain its test size and meaning. Name your actual contribution. Do not claim clinical validation, original data collection, a deployment that does not exist, or experiments that were not run.
