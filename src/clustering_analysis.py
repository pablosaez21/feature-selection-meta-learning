"""Clustering analysis over the generated meta-dataset."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from .config import NOMBRES_SELECTORES, SEED
from .preprocessing import imputar_y_escalar, preparar_meta_features

os.environ.setdefault("OMP_NUM_THREADS", "1")


def calcular_clustering_k3(
    meta_dataset_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, KMeans, PCA]:
    """Scale meta-features, cluster them with k=3 and project with PCA."""
    X_raw, y_label, _ = preparar_meta_features(meta_dataset_df, max_missing_ratio=0.50)
    X_sc = imputar_y_escalar(X_raw)

    km = KMeans(n_clusters=3, random_state=seed, n_init=20)
    labels = km.fit_predict(X_sc.values)

    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X_sc.values)
    return X_sc, y_label, labels, X_2d, km, pca


def analizar_gb_por_cluster(
    meta_dataset_df: pd.DataFrame,
    labels: np.ndarray,
    y_pred_gb_top30: list[str],
) -> pd.DataFrame:
    """Return GB Top-30 accuracy by cluster."""
    df_an = meta_dataset_df.copy()
    df_an["cluster"] = labels
    df_an["pred_gb"] = y_pred_gb_top30
    df_an["correcto_gb"] = [
        r == p for r, p in zip(df_an["best_selector"], df_an["pred_gb"])
    ]

    info_clusters = {
        0: ("Cluster 0 - f_classif", "MEDIA"),
        1: ("Cluster 1 - mutual_info", "ALTA"),
        2: ("Cluster 2 - ambiguo", "BAJA"),
    }

    filas = []
    for c in [0, 1, 2]:
        sub = df_an[df_an["cluster"] == c]
        nombre, confianza = info_clusters[c]
        filas.append({
            "cluster": c,
            "nombre": nombre,
            "confianza": confianza,
            "n": len(sub),
            "aciertos": int(sub["correcto_gb"].sum()),
            "accuracy": accuracy_score(sub["best_selector"], sub["pred_gb"]) if len(sub) else np.nan,
        })
    return pd.DataFrame(filas)


def caracterizar_clusters(meta_dataset_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize interpretable dataset statistics by cluster."""
    df_clusters = meta_dataset_df.copy()
    df_clusters["cluster"] = labels

    nombres_cluster = {
        0: "C0 - f_classif",
        1: "C1 - mutual_info",
        2: "C2 - ambiguo",
    }

    estadisticas = {
        "nr_inst": "N instancias",
        "nr_attr": "N features",
        "nr_class": "N clases",
        "class_ent": "Entropia de clases (bits)",
        "majority_class_error": "Error clase mayoritaria",
        "class_balance_ratio": "Ratio de balance",
        "nr_num": "N features numericas",
        "nr_cat": "N features categoricas",
    }

    filas = []
    for feat, nombre in estadisticas.items():
        if feat not in df_clusters.columns:
            continue
        fila = {"estadistica": nombre}
        for c in [0, 1, 2]:
            sub = df_clusters[df_clusters["cluster"] == c][feat].dropna()
            fila[nombres_cluster[c]] = f"{sub.mean():.2f} +/- {sub.std():.2f}"
        filas.append(fila)

    return pd.DataFrame(filas).set_index("estadistica")

