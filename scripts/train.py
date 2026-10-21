"""Train, compare and save the churn pipeline.

Run: uv run python scripts/train.py [--config configs/tp02.json]

Steps: load + verify the raw CSV, split by customer, fit the baseline and each
candidate pipeline on train, compare on validation, apply the selection rule
fixed in the config, save the selected fitted pipeline and its metadata.
The test split is NOT evaluated here (see scripts/evaluate_test.py).
Outputs (comparison, metadata copy, sample batch) go to reports/tp02/ unless
--evidence-dir is given.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from churnlab.data import (
    CATEGORICAL_FEATURES,
    EXPECTED_SHA256,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RAW_PATH,
    TARGET,
    TARGET_MAPPING,
    encode_target,
    file_sha256,
    load_raw,
    prepare_features,
)
from churnlab.evaluation import evaluate, positive_scores
from churnlab.models import make_baseline, make_classifier
from churnlab.pipeline import PREPROCESSING_SETTINGS, build_pipeline
from churnlab.provenance import code_identity, environment_identity
from churnlab.splits import allocation_table, split_customers

# Run outputs go to an ignored folder; pass --evidence-dir evidence/tp02 to refresh submitted evidence.
DEFAULT_OUTPUT_DIR = Path("reports/tp02")
ALLOCATION_PATH = Path("data/processed/split_allocation.csv")


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_and_evaluate(name, classifier, X_train, y_train, X_valid, y_valid, threshold):
    pipeline = build_pipeline(classifier)
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    row = {
        "model": name,
        "classifier": type(classifier).__name__,
        "params": json.dumps(classifier.get_params(), sort_keys=True, default=str),
        "fit_seconds": round(fit_seconds, 4),
        **evaluate(pipeline, X_valid, y_valid, threshold),
    }
    return pipeline, row


def select_model(table: pd.DataFrame, selection: dict) -> tuple[str, str]:
    """Apply the rule fixed in the config before looking at results."""
    learned = table[table["model"] != "majority_baseline"]
    metric, tolerance = selection["metric"], selection["tie_tolerance"]
    best = learned[metric].max()
    close = set(learned.loc[learned[metric] >= best - tolerance, "model"])
    chosen = next(name for name in selection["simplicity_order"] if name in close)
    best_name = learned.loc[learned[metric].idxmax(), "model"]
    reason = (
        f"Best validation {metric} = {best:.4f} ({best_name}). Candidates within "
        f"{tolerance}: {sorted(close)}. Simplest of these: {chosen}."
    )
    return chosen, reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tp02.json"))
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    config = load_config(args.config)
    seed, threshold = config["seed"], config["threshold"]

    raw = load_raw()
    splits = split_customers(raw, seed=seed, **config["split"])
    allocation = allocation_table(splits)
    ALLOCATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    allocation.to_csv(ALLOCATION_PATH, index=False)

    X_train, y_train = prepare_features(splits["train"]), encode_target(splits["train"])
    X_valid, y_valid = prepare_features(splits["valid"]), encode_target(splits["valid"])

    fitted, rows = {}, []
    candidates = {"majority_baseline": make_baseline()}
    for name, spec in config["candidates"].items():
        candidates[name] = make_classifier(spec["class"], spec["params"], seed)
    for name, classifier in candidates.items():
        fitted[name], row = fit_and_evaluate(
            name, classifier, X_train, y_train, X_valid, y_valid, threshold
        )
        rows.append(row)
        print(f"{name:22s} AP={row['average_precision']:.4f} recall={row['recall']:.4f} "
              f"F1={row['f1']:.4f} acc={row['accuracy']:.4f} fit={row['fit_seconds']}s")

    table = pd.DataFrame(rows)
    out_dir = args.evidence_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    chosen, reason = select_model(table, config["selection"])
    selected = fitted[chosen]
    print(f"Selected: {chosen}. {reason}")

    artifact = Path(config["artifact_path"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, artifact)

    # A small fixed batch (no target column) used to prove reload agreement.
    sample = splits["valid"].head(20)
    sample_features = sample[[ID_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    sample_features.to_csv(out_dir / "sample_batch.csv", index=False)
    pd.DataFrame({
        ID_COLUMN: sample[ID_COLUMN].to_numpy(),
        "churn_probability": positive_scores(selected, prepare_features(sample)),
    }).to_csv(out_dir / "original_predictions.csv", index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": {
            "path": artifact.as_posix(),
            "sha256": file_sha256(artifact),
            "format": "joblib (trusted artifacts only: loading runs pickle)",
            "content": "fitted sklearn Pipeline: preprocessor + classifier",
        },
        "selected_model": chosen,
        "selection": {**config["selection"], "decision": reason},
        "classifier_params": selected.named_steps["classifier"].get_params(),
        "validation_metrics": table.set_index("model").loc[chosen].drop("params").to_dict(),
        "task": {
            "target": TARGET,
            "target_mapping": TARGET_MAPPING,
            "positive_class": "Churn = Yes (1)",
            "threshold": threshold,
            "score": "predict_proba column for class 1",
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "identifier_not_used": ID_COLUMN,
            "input_types": {
                **{name: "float (blank/missing allowed -> imputed)" for name in NUMERIC_FEATURES},
                **{name: "string (unseen category ignored)" for name in CATEGORICAL_FEATURES},
            },
            "boundary_parsing": "churnlab.data.prepare_features",
        },
        "preprocessing": PREPROCESSING_SETTINGS,
        "data": {
            "source_file": RAW_PATH.as_posix(),
            "sha256": EXPECTED_SHA256,
            "split_seed": seed,
            "split_fractions": {"train": 0.6, **config["split"]},
            "split_sizes": {name: len(part) for name, part in splits.items()},
            "allocation_file": ALLOCATION_PATH.as_posix(),
            "allocation_sha256": file_sha256(ALLOCATION_PATH),
        },
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "code": code_identity(),
        "environment": environment_identity(),
        "test_split": "reserved; not used for selection",
    }
    text = json.dumps(metadata, indent=2, default=str)
    artifact.with_suffix(".metadata.json").write_text(text, encoding="utf-8")
    (out_dir / "model_metadata.json").write_text(text, encoding="utf-8")
    allocation.to_csv(out_dir / "split_allocation.csv", index=False)
    print(f"Saved {artifact} (sha256 {metadata['artifact']['sha256'][:12]}...)")


if __name__ == "__main__":
    main()
