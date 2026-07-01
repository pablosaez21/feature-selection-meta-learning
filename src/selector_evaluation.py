"""Evaluation of univariate feature selectors."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .config import SEED, W_ACC, W_STD, W_TIME
from .preprocessing import prepare_X_y


def local_minmax(values: np.ndarray) -> np.ndarray:
    """Normalize values locally among selectors for one dataset."""
    value_range = values.max() - values.min()
    if value_range == 0:
        return np.full_like(values, 0.5, dtype=float)
    return (values - values.min()) / value_range


def compute_composite_scores(
    results: dict[str, dict[str, float]],
    w_acc: float = W_ACC,
    w_std: float = W_STD,
    w_time: float = W_TIME,
) -> dict[str, float]:
    """Compute the notebook composite score for each selector."""
    names = list(results.keys())

    accuracies = np.array([results[s]["acc_mean"] for s in names])
    stds = np.array([results[s]["acc_std"] for s in names])
    times = np.array([results[s]["time"] for s in names])

    scores = (
        w_acc * local_minmax(accuracies)
        - w_std * local_minmax(stds)
        - w_time * local_minmax(times)
    )
    return dict(zip(names, scores))


def evaluate_selectors_on_dataset(X_df: pd.DataFrame, y, seed: int = SEED):
    """Evaluate chi-square, mutual information and ANOVA F-test with StratifiedKFold."""
    X, y = prepare_X_y(X_df, y)
    k = max(2, int(np.sqrt(X.shape[1])))

    selector_config = {
        "chi2": {"score_func": chi2, "use_minmax": True},
        "mutual_info": {"score_func": mutual_info_classif, "use_minmax": False},
        "f_classif": {"score_func": f_classif, "use_minmax": False},
    }

    min_class_samples = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_class_samples)
    if n_splits < 2:
        raise ValueError(f"Minority class has only {min_class_samples} sample(s)")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}

    for selector_name, config in selector_config.items():
        try:
            steps = []
            if config["use_minmax"]:
                steps.append(("scaler", MinMaxScaler()))
            steps.extend([
                ("selector", SelectKBest(score_func=config["score_func"], k=k)),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=seed)),
            ])
            pipe = Pipeline(steps)

            start = time.time()
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            results[selector_name] = {
                "acc_mean": scores.mean(),
                "acc_std": scores.std(),
                "time": time.time() - start,
            }
        except Exception as exc:
            print(f"  {selector_name} failed on this dataset: {exc}")
            results[selector_name] = {"acc_mean": 0.0, "acc_std": 1.0, "time": 1e6}

    composite_scores = compute_composite_scores(results)
    best_selector = max(composite_scores, key=composite_scores.get)
    return composite_scores, best_selector

