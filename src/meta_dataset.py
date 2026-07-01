"""Meta-dataset construction from OpenML tasks."""

from __future__ import annotations

import gc

import openml
import pandas as pd

from .config import MAX_FEATURES, MAX_INSTANCES, SELECTOR_NAMES, SEED
from .data_loading import get_suite_task_ids
from .meta_features import extract_meta_features_pymfe
from .preprocessing import impute_dataframe
from .selector_evaluation import evaluate_selectors_on_dataset


def process_openml_task(task_id: int, seed: int = SEED) -> dict[str, object] | None:
    """Load one OpenML task and return one row for the meta-dataset."""
    try:
        task = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
        target_name = dataset.default_target_attribute or getattr(task, "target_name", None)
        if target_name is None:
            return None

        X_df, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
        y = pd.Series(y)

        if X_df.shape[0] > MAX_INSTANCES or X_df.shape[1] > MAX_FEATURES:
            print(f"  Skipped by size: {dataset.name}")
            return None

        mask = y.notna().values
        X_df = X_df.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        X_df = impute_dataframe(X_df)

        composite_scores, best_selector = evaluate_selectors_on_dataset(X_df, y, seed=seed)
        meta = extract_meta_features_pymfe(X_df, y)

        row: dict[str, object] = {
            "dataset_name": f"{dataset.name}_{task_id}",
            "best_selector": best_selector,
        }
        row.update(meta)
        for selector in SELECTOR_NAMES:
            row[f"score_{selector}"] = composite_scores[selector]
        return row

    except Exception as exc:
        print(f"  Error in task {task_id}: {exc}")
        return None

    finally:
        try:
            del X_df, y
        except Exception:
            pass
        gc.collect()


def build_meta_dataset(task_ids: list[int] | None = None, seed: int = SEED) -> pd.DataFrame:
    """Build the meta-dataset by processing OpenML tasks."""
    task_ids = get_suite_task_ids() if task_ids is None else task_ids
    rows = []

    for i, task_id in enumerate(task_ids, start=1):
        print(f"[{i}/{len(task_ids)}] task_id={task_id}")
        row = process_openml_task(task_id, seed=seed)
        if row is not None:
            rows.append(row)
            print(f"  {row['dataset_name']} - best selector: {row['best_selector']}")

    meta_dataset_df = pd.DataFrame(rows)
    print(f"Meta-dataset: {meta_dataset_df.shape[0]} datasets x {meta_dataset_df.shape[1]} columns")
    return meta_dataset_df

