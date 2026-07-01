"""Preprocessing helpers used by the notebook pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import SELECTOR_NAMES


def impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable columns and impute numeric/categorical missing values."""
    df = df.copy()

    all_nan_cols = df.columns[df.isna().all()]
    if len(all_nan_cols) > 0:
        df = df.drop(columns=all_nan_cols)

    nan_ratio = df.isna().mean()
    almost_nan_cols = nan_ratio[nan_ratio > 0.95].index
    if len(almost_nan_cols) > 0:
        df = df.drop(columns=almost_nan_cols)

    for col in df.columns:
        series = df[col]
        if is_numeric_dtype(series):
            df[col] = series.fillna(float(series.median()))
        else:
            mode = series.mode(dropna=True)
            df[col] = series.fillna(mode.iloc[0])

    return df


def prepare_X_y(X_df: pd.DataFrame, y: pd.Series | np.ndarray):
    """Encode predictors and target as required by scikit-learn selectors."""
    X = pd.get_dummies(X_df, drop_first=False)

    y = pd.Series(y).copy()
    if (
        y.dtype == "object"
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
    ):
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y


def split_meta_dataset(
    meta_dataset_df: pd.DataFrame,
    max_missing_ratio: float | None = None,
):
    """Return meta-feature matrix, labels and dataset names."""
    score_cols = [f"score_{selector}" for selector in SELECTOR_NAMES]
    excluded_cols = ["dataset_name", "best_selector"] + score_cols

    X = meta_dataset_df.drop(columns=excluded_cols, errors="ignore").copy()
    X = X.dropna(axis=1, how="all")
    if max_missing_ratio is not None:
        X = X.loc[:, X.isnull().mean() < max_missing_ratio]

    y = meta_dataset_df["best_selector"].values
    names = meta_dataset_df["dataset_name"].tolist()
    return X, y, names


def impute_and_scale(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute and standardize a meta-feature matrix."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns)

