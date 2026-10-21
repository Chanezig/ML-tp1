"""Reproducible stratified train/validation/test split by customer."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churnlab.data import ID_COLUMN, TARGET

SPLIT_NAMES = ("train", "valid", "test")


def split_customers(
    df: pd.DataFrame, seed: int, valid_size: float = 0.2, test_size: float = 0.2
) -> dict[str, pd.DataFrame]:
    """Split ~60/20/20, stratified on the target, with a single seed."""
    train_df, remaining_df = train_test_split(
        df, test_size=valid_size + test_size, stratify=df[TARGET], random_state=seed
    )
    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=test_size / (valid_size + test_size),
        stratify=remaining_df[TARGET],
        random_state=seed,
    )
    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    check_disjoint(splits)
    return splits


def check_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    seen: set[str] = set()
    for name, part in splits.items():
        ids = set(part[ID_COLUMN])
        overlap = seen & ids
        if overlap:
            raise ValueError(f"{len(overlap)} customer IDs of {name} appear in another split")
        seen |= ids


def allocation_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per customer: customerID, split. Sorted for a stable file."""
    rows = [
        pd.DataFrame({ID_COLUMN: part[ID_COLUMN].to_numpy(), "split": name})
        for name, part in splits.items()
    ]
    return pd.concat(rows).sort_values(ID_COLUMN).reset_index(drop=True)


def splits_from_allocation(df: pd.DataFrame, allocation: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rebuild the same customer allocation from a saved table."""
    split_of = allocation.set_index(ID_COLUMN)["split"]
    assigned = df[ID_COLUMN].map(split_of)
    if assigned.isna().any():
        raise ValueError("Some customers are missing from the allocation table.")
    return {name: df[assigned.eq(name)] for name in SPLIT_NAMES}
