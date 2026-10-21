"""Load the raw Telco snapshot and parse it at the input boundary.

Parsing rules (stateless, applied identically at training and inference):

- Numeric features are converted explicitly to float. A blank or
  whitespace-only value (and an existing NaN/None) becomes missing (NaN);
  missing values are then imputed inside the fitted pipeline.
- Any other non-numeric text (e.g. "abc") raises ValueError instead of
  silently becoming a missing value.
- Categorical features are kept as stripped text; blank values become missing.
- A required feature column that is absent raises ValueError.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/telco.csv")
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

ID_COLUMN = "customerID"
TARGET = "Churn"
POSITIVE_LABEL = "Yes"
TARGET_MAPPING = {"No": 0, "Yes": 1}

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path = RAW_PATH, expected_sha256: str = EXPECTED_SHA256) -> pd.DataFrame:
    """Read the raw CSV as text after checking its hash. The file is never modified."""
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Source hash mismatch for {path}: {actual}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {ID_COLUMN, TARGET, *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate customer IDs in the source file.")
    return df


def parse_numeric(values: pd.Series) -> pd.Series:
    """Convert to float; blank -> NaN; malformed non-blank text -> ValueError."""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype("float64")
    text = values.astype("object").where(values.notna(), None)
    stripped = text.map(lambda v: v.strip() if isinstance(v, str) else v)
    blank = stripped.isna() | stripped.eq("")
    converted = pd.to_numeric(stripped.mask(blank), errors="coerce")
    malformed = converted.isna() & ~blank
    if malformed.any():
        examples = stripped[malformed].unique()[:5].tolist()
        raise ValueError(f"Column {values.name!r} has non-numeric values: {examples}")
    return converted.astype("float64")


def parse_categorical(values: pd.Series) -> pd.Series:
    text = values.astype("object").where(values.notna(), np.nan)
    stripped = text.map(lambda v: v.strip() if isinstance(v, str) else v)
    return stripped.mask(stripped.eq(""), np.nan)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the six documented features and parse them. Never reads the target."""
    missing = [name for name in FEATURES if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    features = pd.DataFrame(index=df.index)
    for name in NUMERIC_FEATURES:
        features[name] = parse_numeric(df[name])
    for name in CATEGORICAL_FEATURES:
        features[name] = parse_categorical(df[name])
    return features


def encode_target(df: pd.DataFrame) -> pd.Series:
    """Churn = Yes is the positive class (1); No is 0."""
    unknown = sorted(set(df[TARGET]) - set(TARGET_MAPPING))
    if unknown:
        raise ValueError(f"Unexpected target labels: {unknown}")
    return df[TARGET].map(TARGET_MAPPING).astype("int64")
