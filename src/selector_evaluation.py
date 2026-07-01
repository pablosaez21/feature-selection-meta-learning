"""Evaluation of univariate feature selectors on one dataset."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .config import NOMBRES_SELECTORES, SEED, W_ACC, W_STD, W_TIEMPO
from .preprocessing import preparar_X_e_y


def _normalizar_local(valores: np.ndarray) -> np.ndarray:
    """Local min-max normalization among selectors from the same dataset."""
    rango = valores.max() - valores.min()
    if rango == 0:
        return np.full_like(valores, 0.5, dtype=float)
    return (valores - valores.min()) / rango


def calcular_scores_compuestos(
    resultados: dict[str, dict[str, float]],
    w_acc: float = W_ACC,
    w_std: float = W_STD,
    w_tiempo: float = W_TIEMPO,
) -> dict[str, float]:
    """Compute the weighted selector score used in the notebook."""
    nombres = list(resultados.keys())

    accs = np.array([resultados[s]["acc_media"] for s in nombres])
    stds = np.array([resultados[s]["acc_std"] for s in nombres])
    tiempos = np.array([resultados[s]["tiempo"] for s in nombres])

    norm_acc = _normalizar_local(accs)
    norm_std = _normalizar_local(stds)
    norm_tiempo = _normalizar_local(tiempos)

    scores_arr = w_acc * norm_acc - w_std * norm_std - w_tiempo * norm_tiempo
    return dict(zip(nombres, scores_arr))


def evaluar_selectores_en_dataset(
    X_df: pd.DataFrame,
    y: pd.Series | np.ndarray,
    seed: int = SEED,
) -> tuple[dict[str, float], str]:
    """Evaluate chi2, mutual_info and f_classif with StratifiedKFold."""
    X, y = preparar_X_e_y(X_df, y)
    k = max(2, int(np.sqrt(X.shape[1])))

    selectores_config = {
        "chi2": {"score_func": chi2, "usar_minmax": True},
        "mutual_info": {"score_func": mutual_info_classif, "usar_minmax": False},
        "f_classif": {"score_func": f_classif, "usar_minmax": False},
    }

    resultados: dict[str, dict[str, float]] = {}
    min_samples_clase = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_samples_clase)
    if n_splits < 2:
        raise ValueError(f"Clase minoritaria con solo {min_samples_clase} muestra/s")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for nombre_selector, config in selectores_config.items():
        try:
            if config["usar_minmax"]:
                pipe = Pipeline([
                    ("scaler", MinMaxScaler()),
                    ("selector", SelectKBest(score_func=config["score_func"], k=k)),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=seed)),
                ])
            else:
                pipe = Pipeline([
                    ("selector", SelectKBest(score_func=config["score_func"], k=k)),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=seed)),
                ])

            t_inicio = time.time()
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            resultados[nombre_selector] = {
                "acc_media": scores.mean(),
                "acc_std": scores.std(),
                "tiempo": time.time() - t_inicio,
            }
        except Exception as exc:
            print(f"  ! {nombre_selector} fallo en este dataset: {exc}")
            resultados[nombre_selector] = {
                "acc_media": 0.0,
                "acc_std": 1.0,
                "tiempo": 1e6,
            }

    scores_compuestos = calcular_scores_compuestos(resultados)
    mejor_selector = max(scores_compuestos, key=scores_compuestos.get)
    return scores_compuestos, mejor_selector

