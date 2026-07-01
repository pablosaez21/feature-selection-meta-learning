"""Clustering analysis over the meta-dataset."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from .config import SEED
from .preprocessing import impute_and_scale, split_meta_dataset

os.environ.setdefault("OMP_NUM_THREADS", "1")


def compute_kmeans_clustering(meta_dataset_df: pd.DataFrame, seed: int = SEED):
    """Cluster scaled meta-features with KMeans k=3 and project them with PCA."""
    X_raw, y_label, _ = split_meta_dataset(meta_dataset_df, max_missing_ratio=0.50)
    X_scaled = impute_and_scale(X_raw)

    kmeans = KMeans(n_clusters=3, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(X_scaled.values)

    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X_scaled.values)
    return X_scaled, y_label, labels, X_2d, kmeans, pca


def analyze_gb_accuracy_by_cluster(
    meta_dataset_df: pd.DataFrame,
    labels: np.ndarray,
    gb_predictions: list[str],
) -> pd.DataFrame:
    """Compute GradientBoosting Top-30 accuracy for each cluster."""
    df = meta_dataset_df.copy()
    df["cluster"] = labels
    df["gb_prediction"] = gb_predictions
    df["gb_correct"] = [r == p for r, p in zip(df["best_selector"], df["gb_prediction"])]

    cluster_info = {
        0: ("Cluster 0 - f_classif", "medium"),
        1: ("Cluster 1 - mutual_info", "high"),
        2: ("Cluster 2 - ambiguous", "low"),
    }

    rows = []
    for cluster_id in [0, 1, 2]:
        subset = df[df["cluster"] == cluster_id]
        name, confidence = cluster_info[cluster_id]
        rows.append({
            "cluster": cluster_id,
            "name": name,
            "confidence": confidence,
            "n_datasets": len(subset),
            "correct": int(subset["gb_correct"].sum()),
            "accuracy": accuracy_score(subset["best_selector"], subset["gb_prediction"]) if len(subset) else np.nan,
        })
    return pd.DataFrame(rows)


def characterize_clusters(meta_dataset_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize interpretable dataset statistics by cluster."""
    df = meta_dataset_df.copy()
    df["cluster"] = labels

    cluster_names = {
        0: "C0 - f_classif",
        1: "C1 - mutual_info",
        2: "C2 - ambiguous",
    }
    statistics = {
        "nr_inst": "Number of instances",
        "nr_attr": "Number of features",
        "nr_class": "Number of classes",
        "class_ent": "Class entropy",
        "majority_class_error": "Majority-class error",
        "class_balance_ratio": "Class balance ratio",
        "nr_num": "Number of numerical features",
        "nr_cat": "Number of categorical features",
    }

    rows = []
    for feature, label in statistics.items():
        if feature not in df.columns:
            continue
        row = {"statistic": label}
        for cluster_id in [0, 1, 2]:
            subset = df[df["cluster"] == cluster_id][feature].dropna()
            row[cluster_names[cluster_id]] = f"{subset.mean():.2f} +/- {subset.std():.2f}"
        rows.append(row)

    return pd.DataFrame(rows).set_index("statistic")

