"""Unfitted classifiers built from the training configuration."""

import inspect

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    "DummyClassifier": DummyClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}


def make_classifier(class_name: str, params: dict, seed: int):
    """Instantiate a classifier; random_state is set to the seed when supported."""
    cls = CLASSIFIERS[class_name]
    params = dict(params)
    if "random_state" in inspect.signature(cls).parameters:
        params.setdefault("random_state", seed)
    return cls(**params)


def make_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")
