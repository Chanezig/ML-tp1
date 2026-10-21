"""Build unfitted preprocessing + classifier pipelines.

These functions only construct objects: they never read data or fit anything.
All learned statistics (medians, means, scales, categories) are learned when
the returned pipeline is fitted, i.e. on training data only.

Preprocessing policy:
- numeric: median imputation, then standardisation (StandardScaler);
- categorical: missing -> constant "missing", then one-hot encoding;
  a category never seen in training is ignored (all-zero encoding).
"""

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlab.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PREPROCESSING_SETTINGS = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_imputer": "median",
    "numeric_scaler": "StandardScaler",
    "categorical_imputer": "constant ('missing')",
    "categorical_encoder": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
    "unknown_category_policy": "ignored: encoded as all zeros for that feature",
}


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_pipeline(classifier) -> Pipeline:
    """Fresh preprocessor + an unfitted copy of the classifier."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", clone(classifier)),
    ])
