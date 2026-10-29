"""Read only the original processed Cleveland schema; reject ambiguous encodings."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heart.config import BOUNDS, CATEGORIES, FEATURES, RAW_PATH, ROOT, SEED, TEST_SIZE


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_features(frame):
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    if set(frame.columns) != set(FEATURES):
        missing = sorted(set(FEATURES) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(FEATURES))
        raise ValueError(f"Feature columns do not match. Missing: {missing}; extra: {extra}")
    if frame.empty:
        raise ValueError("Supply at least one input row.")
    result = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite values are not supported.")
    if result.isna().all(axis=1).any():
        raise ValueError("A row cannot have all features missing.")
    for name, mapping in CATEGORIES.items():
        if not result[name].dropna().isin(mapping).all():
            raise ValueError(f"{name} must use the original UCI codes: {list(mapping)}")
    for name, (low, high) in BOUNDS.items():
        if not result[name].dropna().between(low, high).all():
            raise ValueError(f"{name} must be between {low} and {high}.")
    if not result["ca"].dropna().isin([0, 1, 2, 3]).all():
        raise ValueError("ca must be a whole-number count from 0 to 3.")
    return result


def load_dataset(path=RAW_PATH, verify=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run: python -m heart.download")
    provenance = ROOT / "data" / "provenance.json"
    if verify and provenance.exists():
        expected = json.loads(provenance.read_text(encoding="utf-8"))["raw_sha256"]
        if sha256(path) != expected:
            raise ValueError("Dataset checksum differs from the documented Cleveland source.")
    raw = pd.read_csv(path, header=None, na_values="?")
    if raw.shape[1] != 14:
        raise ValueError("Expected 14 columns in processed.cleveland.data (no header).")
    raw.columns = FEATURES + ["num"]
    y = pd.to_numeric(raw.pop("num"), errors="raise")
    if y.isna().any() or not y.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("num must contain original UCI target labels 0 through 4.")
    X = validate_features(raw)
    if X.duplicated().any():
        raise ValueError("Duplicate feature rows detected. Investigate before splitting.")
    return X, (y > 0).astype(int).rename("target")


def split_dataset(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
