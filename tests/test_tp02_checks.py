"""TP02 focused reliability checks. Run from the project root: uv run pytest -v"""

import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churnlab.data import (
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RAW_PATH,
    encode_target,
    load_raw,
    prepare_features,
)
from churnlab.evaluation import positive_scores
from churnlab.pipeline import build_pipeline
from churnlab.splits import allocation_table, split_customers

SEED = 42

pytestmark = pytest.mark.skipif(
    not RAW_PATH.exists(), reason="run scripts/download_data.py first"
)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def splits(raw):
    return split_customers(raw, seed=SEED)


@pytest.fixture(scope="module")
def fitted(splits):
    train = splits["train"]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(prepare_features(train), encode_target(train))
    return pipeline


def numeric_imputer(pipeline):
    return pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]


def test_customer_ids_disjoint_and_complete(raw, splits):
    ids = {name: set(part[ID_COLUMN]) for name, part in splits.items()}
    assert not ids["train"] & ids["valid"]
    assert not ids["train"] & ids["test"]
    assert not ids["valid"] & ids["test"]
    assert ids["train"] | ids["valid"] | ids["test"] == set(raw[ID_COLUMN])
    assert [len(splits[n]) for n in ("train", "valid", "test")] == [4225, 1409, 1409]


def test_split_is_reproducible_and_stratified(raw, splits):
    again = allocation_table(split_customers(raw, seed=SEED))
    assert again.equals(allocation_table(splits))
    rates = [encode_target(part).mean() for part in splits.values()]
    assert max(rates) - min(rates) < 0.005


def test_imputer_medians_come_from_training_only(raw, splits, fitted):
    learned = numeric_imputer(fitted).statistics_
    train_medians = prepare_features(splits["train"])[NUMERIC_FEATURES].median().to_numpy()
    full_medians = prepare_features(raw)[NUMERIC_FEATURES].median().to_numpy()
    np.testing.assert_allclose(learned, train_medians)
    # TotalCharges median differs between train and full data, so this is discriminating.
    assert not np.allclose(learned, full_medians)


def test_prediction_does_not_refit(splits, fitted):
    imputer = numeric_imputer(fitted)
    scaler = fitted.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    before = (imputer.statistics_.copy(), scaler.mean_.copy(), scaler.scale_.copy())
    fitted.predict(prepare_features(splits["valid"]))
    np.testing.assert_array_equal(imputer.statistics_, before[0])
    np.testing.assert_array_equal(scaler.mean_, before[1])
    np.testing.assert_array_equal(scaler.scale_, before[2])


def test_blank_numeric_is_imputed_with_training_median(fitted):
    record = pd.DataFrame([{
        "tenure": "0", "MonthlyCharges": "50.0", "TotalCharges": " ",
        "Contract": "Month-to-month", "InternetService": "DSL",
        "PaymentMethod": "Electronic check",
    }])
    X = prepare_features(record)
    assert np.isnan(X.loc[0, "TotalCharges"])
    numeric = fitted.named_steps["preprocessor"].named_transformers_["num"]
    imputed = numeric.named_steps["imputer"].transform(X[NUMERIC_FEATURES])
    assert imputed[0, 2] == numeric_imputer(fitted).statistics_[2]
    assert np.isfinite(positive_scores(fitted, X)).all()


def test_unseen_category_is_encoded_as_zeros(fitted):
    record = pd.DataFrame([{
        "tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0,
        "Contract": "Five year", "InternetService": "DSL",
        "PaymentMethod": "Electronic check",
    }])
    preprocessor = fitted.named_steps["preprocessor"]
    encoded = preprocessor.transform(prepare_features(record))
    names = preprocessor.get_feature_names_out()
    contract_columns = [i for i, n in enumerate(names) if n.startswith("cat__Contract_")]
    assert encoded[0, contract_columns].sum() == 0
    # A score is produced, but nothing guarantees it is reliable for this customer.
    assert np.isfinite(positive_scores(fitted, prepare_features(record))).all()


def test_malformed_numeric_text_fails_visibly():
    record = pd.DataFrame([{
        "tenure": "12", "MonthlyCharges": "abc", "TotalCharges": "100",
        "Contract": "One year", "InternetService": "DSL", "PaymentMethod": "Mailed check",
    }])
    with pytest.raises(ValueError, match="non-numeric"):
        prepare_features(record)


def test_missing_required_feature_fails_visibly(splits, fitted):
    batch = splits["valid"].head(5).drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Missing required feature"):
        prepare_features(batch)
    # Even if the boundary check were bypassed, the pipeline itself refuses.
    X = prepare_features(splits["valid"].head(5)).drop(columns=["Contract"])
    with pytest.raises(ValueError):
        fitted.predict(X)


def test_reload_in_fresh_process_preserves_predictions(splits, fitted, tmp_path):
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(fitted, artifact)
    batch = splits["valid"].head(20)[FEATURES]  # no Churn column
    batch_path = tmp_path / "batch.csv"
    batch.to_csv(batch_path, index=False)
    expected = positive_scores(fitted, prepare_features(batch))

    code = (
        "import sys, joblib, pandas as pd\n"
        "from churnlab.data import prepare_features\n"
        "from churnlab.evaluation import positive_scores\n"
        "p = joblib.load(sys.argv[1])\n"
        "b = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)\n"
        "print('\\n'.join(repr(float(s)) for s in positive_scores(p, prepare_features(b))))\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code, str(artifact), str(batch_path)], text=True
    )
    reloaded = np.array([float(line) for line in out.split()])
    np.testing.assert_allclose(reloaded, expected, rtol=0, atol=1e-12)
