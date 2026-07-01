"""Preprocessing helpers used by the meta-learning pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import NOMBRES_SELECTORES


def imputar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable columns and impute numeric/categorical missing values."""
    df = df.copy()

    cols_all_nan = df.columns[df.isna().all()]
    if len(cols_all_nan) > 0:
        df = df.drop(columns=cols_all_nan)

    pct_nan = df.isna().mean()
    cols_casi_nan = pct_nan[pct_nan > 0.95].index
    if len(cols_casi_nan) > 0:
        df = df.drop(columns=cols_casi_nan)

    for col in df.columns:
        serie = df[col]
        if is_numeric_dtype(serie):
            df[col] = serie.fillna(float(serie.median()))
        else:
            moda = serie.mode(dropna=True)
            df[col] = serie.fillna(moda.iloc[0])

    return df


def preparar_X_e_y(X_df: pd.DataFrame, y: pd.Series | np.ndarray) -> tuple[pd.DataFrame, pd.Series | np.ndarray]:
    """Encode predictors and target as required by sklearn selectors."""
    X = pd.get_dummies(X_df, drop_first=False)

    y = pd.Series(y).copy()
    if (
        y.dtype == "object"
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
    ):
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y


def preparar_meta_features(
    meta_dataset_df: pd.DataFrame,
    max_missing_ratio: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build X/y for meta-learners, excluding scores and labels."""
    score_cols = [f"score_{sel}" for sel in NOMBRES_SELECTORES]
    cols_excluir = ["dataset_name", "best_selector"] + score_cols

    X = meta_dataset_df.drop(columns=cols_excluir, errors="ignore").copy()
    X = X.dropna(axis=1, how="all")
    if max_missing_ratio is not None:
        X = X.loc[:, X.isnull().mean() < max_missing_ratio]

    y = meta_dataset_df["best_selector"].values
    dataset_names = meta_dataset_df["dataset_name"].tolist()
    return X, y, dataset_names


def imputar_y_escalar(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute and standardize a meta-feature matrix."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns)

