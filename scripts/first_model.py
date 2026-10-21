"""Milestone 1: a first model on tenure + MonthlyCharges, with a majority baseline.

Run: uv run python scripts/first_model.py
"""

import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from churnlab.data import encode_target, load_raw, parse_numeric
from churnlab.splits import split_customers

config = json.loads(Path("configs/tp02.json").read_text(encoding="utf-8"))
splits = split_customers(load_raw(), seed=config["seed"], **config["split"])

first_features = ["tenure", "MonthlyCharges"]


def xy(part):
    X = part[first_features].apply(parse_numeric)
    return X, encode_target(part)


X_train, y_train = xy(splits["train"])
X_valid, y_valid = xy(splits["valid"])

# Both models are fitted on training data only; validation is only predicted.
models = {
    "majority_baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression_2_features": LogisticRegression(max_iter=1000),
}

report = {
    "features": first_features,
    "positive_class": "Churn = Yes (1)",
    "confusion_matrix_layout": "rows = true [No(0), Yes(1)], columns = predicted [No(0), Yes(1)]",
    "seed": config["seed"],
    "split_sizes": {name: len(part) for name, part in splits.items()},
    "results": {},
}
for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_valid)
    report["results"][name] = {
        "valid_accuracy": float(accuracy_score(y_valid, predicted)),
        "valid_confusion_matrix": confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist(),
        "predicted_churners": int(predicted.sum()),
    }

out = Path("evidence/tp02/first_model.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
