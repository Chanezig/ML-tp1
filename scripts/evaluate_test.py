"""Final, single evaluation of the already-selected pipeline on the reserved test split.

Run once, after selection is finished: uv run python scripts/evaluate_test.py
The result is a held-out estimate; it must not be used to change the model choice.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from churnlab.data import encode_target, load_raw, prepare_features
from churnlab.evaluation import evaluate
from churnlab.splits import splits_from_allocation

artifact = Path("artifacts/churn_pipeline.joblib")
metadata = json.loads(artifact.with_suffix(".metadata.json").read_text(encoding="utf-8"))
allocation = pd.read_csv(metadata["data"]["allocation_file"], dtype=str)
test_df = splits_from_allocation(load_raw(), allocation)["test"]

pipeline = joblib.load(artifact)
report = {
    "selected_model": metadata["selected_model"],
    "artifact_sha256": metadata["artifact"]["sha256"],
    "test_rows": len(test_df),
    "note": "Single held-out evaluation after selection; not used for model choice.",
    **evaluate(pipeline, prepare_features(test_df), encode_target(test_df), metadata["task"]["threshold"]),
}
out = Path("evidence/tp02/test_evaluation.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
