from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import CATEGORIES, NUMERIC, SEED


def make_pipeline(estimator):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(categories=[list(v) for v in CATEGORIES.values()],
                                 handle_unknown="error", sparse_output=False)),
    ])
    transform = ColumnTransformer([("numeric", numeric, NUMERIC),
                                   ("categorical", categorical, list(CATEGORIES))])
    return Pipeline([("preprocess", transform), ("model", estimator)])


def candidates(include_xgboost=True):
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5,
                                                min_samples_leaf=3, random_state=SEED, n_jobs=1),
    }
    if include_xgboost:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.03,
                                          subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
                                          objective="binary:logistic", eval_metric="logloss",
                                          random_state=SEED, n_jobs=1)
    return {name: make_pipeline(model) for name, model in models.items()}
