# Start here

## What is already included

The complete Python source, original public dataset, a trained model, real evaluation results, figures, an exploration notebook, tests, and deployment instructions are in this folder. There is no API key, paid service, database, GPU, or Kaggle login to configure.

You still need to install Python dependencies on your computer. Python itself and its installed packages are not included in the ZIP. Initial setup needs internet access; the local app can subsequently use the bundled files offline.

## 1. Extract the ZIP

Extract all files. Open the inner `heart-disease-classifier` folder in VS Code, then choose **Terminal > New Terminal**. Check that `app.py` and `requirements.txt` are directly inside the current folder. Do not run from inside the ZIP viewer.

## 2. Install Python 3.12

This package was tested on Python 3.12. Use a Python 3.12 installation from [python.org](https://www.python.org/downloads/). On Windows, enable the Python launcher and PATH option if the installer offers them, then reopen your terminal.

Check:

```powershell
py -3.12 --version
```

If `py` is not recognized but `python --version` shows 3.12, replace only the first `py -3.12` below with `python`. If your existing Python is 3.14 or another version, install 3.12 alongside it and use `py -3.12` explicitly.

## 3. Launch on Windows

From the folder containing `app.py`, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Using the full virtual-environment path avoids PowerShell activation-policy problems. Dependencies take several minutes to download the first time. Open the local URL printed in the terminal, usually `http://localhost:8501`. Keep the terminal running. Stop with **Ctrl+C**.

Later launches need just:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## macOS / Linux alternative

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

If XGBoost on macOS reports `libomp` missing, install the OpenMP runtime using `brew install libomp` if you use Homebrew. Otherwise use the reduced-model procedure in the troubleshooting section.

## 4. Reproduce the results once

Close the app, then on Windows:

```powershell
.\.venv\Scripts\python.exe -m heart.download
.\.venv\Scripts\python.exe -m heart.explore
.\.venv\Scripts\python.exe -m heart.train
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python` instead. These commands recreate the reports and saved model. Training uses all four candidates and normally takes a few minutes on a laptop. Minor numeric differences can occur across platforms.

Open `reports/evaluation.md` to read your results. Open `docs/PROJECT_WALKTHROUGH.md` to understand the code and prepare your explanation. You do not have to complete a notebook to launch the app.

## 5. Optional notebook

Install the Jupyter extension in VS Code, then:

```powershell
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Open `notebooks/01_exploration.ipynb`, select the `.venv` interpreter, and run the cells from top to bottom. The notebook is supplied without saved cell outputs; the equivalent generated figures are already in `reports/figures`.

## 6. Put it on GitHub and deploy

Read `docs/DEPLOYMENT.md`. You must create/sign into your own GitHub and Streamlit accounts, publish the repository, and select its `app.py` entry point. No account actions have been performed for you.

## 7. Before adding it to your resume

Run the app and training yourself. Read the evaluation report, understand the split and preprocessing, and explain one failure case. Add a small documented experiment of your own using training cross-validation. Replace any resume metric with the result from your saved run. Describe your contribution accurately and follow any applicable disclosure rules for coursework or hiring assessments.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named ...` | Install requirements using the same `.venv` Python used to launch the app. |
| Cannot find `app.py` | Change the terminal folder to the extracted folder containing `app.py`. |
| Model version warning or artifact mismatch | Use the pinned requirements, close the app, run `python -m heart.train` with the environment's Python, and restart. |
| Port 8501 is busy | Append `--server.port 8502` to the Streamlit command. |
| Missing raw dataset | Run `python -m heart.download`; it retrieves and verifies the original source. |
| Certificate/proxy/download error | Use your normal permitted internet connection. Do not disable TLS certificate verification. The raw dataset is already bundled. |
| XGBoost cannot install/load | Copy `requirements.txt` to a local temporary requirements file, remove just the XGBoost line, install the remaining packages, then run `python -m heart.train --skip-xgboost`. Remove XGBoost from cloud requirements too if deploying this reduced version. |
| You edited categories and now get a validation error | Restore original UCI codes; read `docs/DATA_DICTIONARY.md`. |

Only load trusted `joblib` model files: that format can execute Python during loading. The app has no model-upload feature.
