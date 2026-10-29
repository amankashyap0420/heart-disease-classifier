# Publish the demo

## Prepare the repository

1. Launch and test the application locally with Python 3.12.
2. Create a GitHub repository, for example `heart-disease-classifier`.
3. Upload the contents of this project folder, not the outer ZIP. `app.py` and `requirements.txt` should be at the repository root.
4. Include `heart/`, `artifacts/`, `data/`, `reports/`, and `.streamlit/config.toml`. Include the docs, tests, notebook and workflow too.
5. Do not upload `.venv`, caches, local secrets or unrelated personal files. `.gitignore` excludes common local files.

If Git is installed, one way to create the initial local commit is:

```bash
git init
git add .
git commit -m "Add Cleveland classification experiment and Streamlit demo"
git branch -M main
```

Then follow the remote/push commands GitHub displays for your new repository. Use your own repository address.

## Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using the GitHub account containing the repository.
2. Choose the option to create/deploy an app, select the repository and branch, and set the main file to `app.py`.
3. Select **Python 3.12** in deployment settings where available. No secrets are needed.
4. Deploy and wait for dependencies to install. The app loads the bundled trained pipeline; it does not retrain on each page view.
5. Test the default sample, an unknown field, the Evaluation tab and the CSV download on the public URL.
6. Add the URL and a screenshot to your README and resume.

If the hosting service's available Python versions change, choose a supported version, recreate and test the environment locally with that version, and retrain before publishing the artifact. Do not ignore scikit-learn artifact-version warnings.

See [the official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) for the current interface. Hosting limits and availability may change.

## Updating a model

Stop the local app before retraining. Run the training command and tests, then commit the artifact, its matching metadata and the generated reports together. The model checksum detects accidental mismatches. Commit `model.joblib` only from your own trusted training process.

The demo does not log or save submitted feature values. Keep demonstrations based on artificial examples; no patient upload workflow is required for this portfolio project.
