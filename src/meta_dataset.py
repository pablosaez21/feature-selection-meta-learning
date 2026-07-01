"""Construction of the meta-dataset."""

from __future__ import annotations

import gc

import openml
import pandas as pd

from .config import MAX_FEATURES, MAX_INSTANCIAS, NOMBRES_SELECTORES, SEED
from .data_loading import obtener_task_ids_suites
from .meta_features import extraer_meta_features_pymfe
from .preprocessing import imputar_dataframe
from .selector_evaluation import evaluar_selectores_en_dataset


def procesar_dataset_completo(task_id: int, seed: int = SEED) -> dict[str, object] | None:
    """Load one OpenML task and return one meta-dataset row."""
    try:
        task = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
        target_name = dataset.default_target_attribute or getattr(task, "target_name", None)
        if target_name is None:
            return None

        X_df, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
        y = pd.Series(y)

        if X_df.shape[0] > MAX_INSTANCIAS or X_df.shape[1] > MAX_FEATURES:
            print(f"  -> Saltado (tamano real): {dataset.name}")
            return None

        mascara = y.notna().values
        X_df = X_df.loc[mascara].reset_index(drop=True)
        y = y.loc[mascara].reset_index(drop=True)
        X_df = imputar_dataframe(X_df)

        nombre = f"{dataset.name}_{task_id}"
        scores_compuestos, mejor_selector = evaluar_selectores_en_dataset(X_df, y, seed=seed)
        meta = extraer_meta_features_pymfe(X_df, y)

        fila: dict[str, object] = {"dataset_name": nombre, "best_selector": mejor_selector}
        fila.update(meta)
        for sel in NOMBRES_SELECTORES:
            fila[f"score_{sel}"] = scores_compuestos[sel]
        return fila

    except Exception as exc:
        print(f"  x Error en task {task_id}: {exc}")
        return None

    finally:
        try:
            del X_df, y
        except Exception:
            pass
        gc.collect()


def construir_meta_dataset(task_ids: list[int] | None = None, seed: int = SEED) -> pd.DataFrame:
    """Build the full meta-dataset by iterating OpenML tasks."""
    if task_ids is None:
        task_ids = obtener_task_ids_suites()

    filas_meta = []
    for i, task_id in enumerate(task_ids, start=1):
        print(f"[{i}/{len(task_ids)}] task_id={task_id}")
        fila = procesar_dataset_completo(task_id, seed=seed)
        if fila is not None:
            filas_meta.append(fila)
            print(f"  ok {fila['dataset_name']} - mejor selector: {fila['best_selector']}")

    meta_dataset_df = pd.DataFrame(filas_meta)
    print(f"\nMeta-dataset final: {meta_dataset_df.shape[0]} datasets x {meta_dataset_df.shape[1]} columnas")
    return meta_dataset_df

