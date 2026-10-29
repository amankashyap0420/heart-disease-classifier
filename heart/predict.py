"""Inference shares training's schema and saved preprocessing."""
import json
from pathlib import Path

import joblib
import pandas as pd

from heart.config import ARTIFACT_DIR, FEATURES
from heart.data import sha256, validate_features


def load_artifact(directory=ARTIFACT_DIR):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    path = directory / "model.joblib"
    if sha256(path) != metadata["model_sha256"]:
        raise ValueError("Model and metadata do not match. Retrain with python -m heart.train.")
    # joblib is pickle-based: only load artifacts from a source you trust.
    model = joblib.load(path)
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError("Saved model uses a different input schema.")
    return model, metadata


def predict_rows(model, frame, threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1.")
    X = validate_features(frame)
    scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return pd.DataFrame({"model_score": scores, "predicted_class": (scores >= threshold).astype(int)},
                        index=X.index)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict a CSV using original UCI feature codes.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    model, metadata = load_artifact()
    frame = pd.read_csv(args.input, na_values=["?"])
    result = predict_rows(model, frame, metadata["threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
