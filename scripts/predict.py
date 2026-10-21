"""Inference only: load the saved fitted pipeline and score new customers.

Run: uv run python scripts/predict.py --input evidence/tp02/sample_batch.csv
     [--model artifacts/churn_pipeline.joblib] [--output reports/predictions.csv]

No fit is called and no target column is needed. The artifact hash is checked
against its metadata before loading (joblib runs pickle: trusted files only).
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from churnlab.data import ID_COLUMN, file_sha256, prepare_features
from churnlab.evaluation import positive_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn_pipeline.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args()

    metadata = json.loads(args.model.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if file_sha256(args.model) != metadata["artifact"]["sha256"]:
        raise ValueError("Artifact hash does not match its metadata: refusing to load.")
    pipeline = joblib.load(args.model)
    threshold = metadata["task"]["threshold"]

    batch = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    scores = positive_scores(pipeline, prepare_features(batch))
    result = pd.DataFrame({
        "churn_probability": scores,
        "churn_predicted": (scores >= threshold).astype(int),
    })
    if ID_COLUMN in batch.columns:
        result.insert(0, ID_COLUMN, batch[ID_COLUMN].to_numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
