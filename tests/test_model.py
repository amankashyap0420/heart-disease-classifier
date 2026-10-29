import joblib
import numpy as np
import pytest

from heart.config import NUMERIC
from heart.data import load_dataset, split_dataset
from heart.modeling import candidates
from heart.predict import load_artifact, predict_rows


def test_pipeline_fits_imputation_on_training_only(tmp_path):
    X, y = load_dataset()
    train, test, ytrain, _ = split_dataset(X, y)
    model = candidates(include_xgboost=False)["logistic_regression"].fit(train, ytrain)
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, train[NUMERIC].median().to_numpy())
    before = imputer.statistics_.copy()
    model.predict_proba(test)
    np.testing.assert_array_equal(imputer.statistics_, before)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(model.predict_proba(test), joblib.load(path).predict_proba(test))


def test_saved_artifact_predictions_match_report():
    import pandas as pd
    from heart.config import REPORT_DIR
    X, _ = load_dataset()
    model, metadata = load_artifact()
    report = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    predicted = predict_rows(model, X.loc[report.source_row], metadata["threshold"])
    np.testing.assert_allclose(predicted.model_score, report.score, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(predicted.predicted_class, report.predicted)


def test_missing_features_are_imputed_and_predictions_are_bounded():
    X, _ = load_dataset()
    model, _ = load_artifact()
    frame = X.iloc[:2].copy()
    frame["ca"] = np.nan
    frame["thal"] = np.nan
    result = predict_rows(model, frame)
    assert result.model_score.between(0, 1).all()
    with pytest.raises(ValueError):
        predict_rows(model, frame, threshold=1.0)
