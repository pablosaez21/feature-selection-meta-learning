"""OpenML task loading helpers."""

from __future__ import annotations

import pandas as pd
import openml

from .config import OPENML_SUITES


def obtener_task_ids_suites(suite_names: list[str] | None = None) -> list[int]:
    """Return the unique task ids from the OpenML suites used in the notebook."""
    suite_names = OPENML_SUITES if suite_names is None else suite_names
    task_ids: set[int] = set()
    for suite_name in suite_names:
        suite = openml.study.get_suite(suite_name)
        task_ids.update(list(suite.tasks))
    return list(task_ids)


def guardar_meta_dataset(meta_dataset_df: pd.DataFrame, base_name: str = "meta_dataset_df2") -> None:
    """Persist the meta-dataset in the two formats used by the notebook."""
    meta_dataset_df.to_pickle(f"{base_name}.pkl")
    meta_dataset_df.to_csv(f"{base_name}.csv", index=False)


def cargar_meta_dataset(path: str = "meta_dataset_df2.pkl") -> pd.DataFrame:
    """Load a previously generated meta-dataset."""
    return pd.read_pickle(path)

