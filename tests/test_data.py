import numpy as np
import pandas as pd
import pytest

from heart.config import FEATURES
from heart.data import load_dataset, split_dataset, validate_features


def sample():
    return pd.DataFrame([[54, 1, 2, 130, 240, 0, 0, 150, 0, 1, 2, 0, 3]], columns=FEATURES)


def test_original_dataset_and_target():
    X, y = load_dataset()
    assert X.shape == (303, 13)
    assert y.value_counts().to_dict() == {0: 164, 1: 139}
    assert int(X.isna().sum().sum()) == 6
    assert not X.duplicated().any()


def test_split_disjoint_complete_and_repeatable():
    X, y = load_dataset()
    a, b, ya, yb = split_dataset(X, y)
    assert set(a.index).isdisjoint(b.index)
    assert set(a.index) | set(b.index) == set(X.index)
    assert len(a) == 242 and len(b) == 61
    assert a.index.equals(ya.index) and b.index.equals(yb.index)
    assert a.equals(split_dataset(X, y)[0])


@pytest.mark.parametrize("column,value", [("cp", 0), ("thal", 2), ("sex", 3), ("ca", 1.5),
                                         ("age", -1), ("chol", np.inf)])
def test_invalid_values_rejected(column, value):
    frame = sample()
    frame[column] = value
    with pytest.raises(ValueError):
        validate_features(frame)


def test_unknown_values_allowed_but_empty_row_rejected():
    frame = sample().astype(float)
    frame.loc[0, "thal"] = np.nan
    assert validate_features(frame).thal.isna().all()
    frame.loc[0, :] = np.nan
    with pytest.raises(ValueError, match="all features missing"):
        validate_features(frame)


def test_column_order_normalized_and_missing_column_rejected():
    frame = sample()
    assert list(validate_features(frame[FEATURES[::-1]]).columns) == FEATURES
    with pytest.raises(ValueError, match="columns do not match"):
        validate_features(frame.drop(columns="age"))
