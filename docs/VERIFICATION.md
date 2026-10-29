# Verification of the supplied package

The included run used Python 3.12 on Windows with the direct dependency versions pinned in `requirements.txt`. The complete installed package list is recorded in `reports/environment.json`.

The package was checked by:

- Training and evaluating all four candidate pipelines on the bundled UCI file.
- Running 16 automated tests covering the data, fixed split, invalid encodings, missing values, training-only imputation, persistence, saved predictions and Streamlit form interactions.
- Running CSV inference through the command-line entry point.
- Executing the exploration notebook's Python cells using a noninteractive plotting backend.
- Generating and inspecting evaluation figures.
- Starting Streamlit locally and checking its HTTP page and health endpoint.

The measured results are in `reports/evaluation.md`. No external clinical validation, cloud deployment or account publication was performed. GitHub Actions and Streamlit deployment instructions are supplied for you to run in your own accounts; Linux/cloud execution is not claimed as locally verified.

The notebook intentionally has no saved outputs. Its figures can be regenerated interactively, and equivalent training-only plots are bundled in `reports/figures`.
