"""OpenML loading and meta-dataset persistence helpers."""

from __future__ import annotations

import pandas as pd
import openml

from .config import OPENML_SUITES


def get_suite_task_ids(suite_names: list[str] | None = None) -> list[int]:
    """Return unique OpenML task IDs from the configured suites."""
    suite_names = OPENML_SUITES if suite_names is None else suite_names
    task_ids: set[int] = set()
    for suite_name in suite_names:
        suite = openml.study.get_suite(suite_name)
        task_ids.update(list(suite.tasks))
    return list(task_ids)


def save_meta_dataset(meta_dataset_df: pd.DataFrame, base_path: str = "results/meta_dataset_df2") -> None:
    """Save the meta-dataset as pickle and CSV."""
    meta_dataset_df.to_pickle(f"{base_path}.pkl")
    meta_dataset_df.to_csv(f"{base_path}.csv", index=False)


def load_meta_dataset(path: str = "results/meta_dataset_df2.pkl") -> pd.DataFrame:
    """Load a previously generated meta-dataset."""
    return pd.read_pickle(path)

