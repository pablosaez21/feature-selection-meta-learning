"""Meta-feature extraction with PyMFE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymfe.mfe import MFE

from .config import MFE_GROUPS


def clean_meta_feature_name(name: object) -> str:
    """Normalize a PyMFE feature name for pandas and scikit-learn."""
    return (
        str(name)
        .replace(".", "_").replace("-", "_").replace(" ", "_")
        .replace("(", "").replace(")", "").replace("/", "_")
        .replace(":", "_").lower()
    )


def to_float(value: object) -> float:
    """Convert a value to float, returning NaN when conversion fails."""
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def categorical_column_indices(df: pd.DataFrame) -> list[int]:
    """Return categorical column indices for PyMFE."""
    return [
        i for i, col in enumerate(df.columns)
        if pd.api.types.is_object_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(df[col])
    ]


def extract_mfe_block(
    X_df: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    summary: list[str] | None = None,
) -> dict[str, float]:
    """Extract one PyMFE block and apply the notebook conversion rules."""
    mfe = MFE(
        groups=MFE_GROUPS,
        features=features,
        summary=summary,
        suppress_warnings=True,
    )
    mfe.fit(X_df.to_numpy(), y.to_numpy(), cat_cols=categorical_column_indices(X_df))
    names, values = mfe.extract()

    block = {}
    for name, value in zip(names, values):
        clean_name = clean_meta_feature_name(name)
        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value).ravel()
            if value.size == 1:
                value = value.item()
            else:
                block[clean_name] = np.nan
                continue
        block[clean_name] = to_float(value)
    return block


def extract_meta_features_pymfe(X_df: pd.DataFrame, y) -> dict[str, float]:
    """Extract the meta-features used in the notebook."""
    X_df = X_df.copy()
    y = pd.Series(y).reset_index(drop=True)
    meta: dict[str, float] = {}

    class_counts = y.value_counts(dropna=False)
    max_class = int(class_counts.max()) if len(class_counts) > 0 else 0
    min_class = int(class_counts.min()) if len(class_counts) > 0 else 0
    meta["class_balance_ratio"] = to_float(min_class / max_class) if max_class > 0 else np.nan
    meta["majority_class_error"] = to_float(1.0 - (max_class / len(y))) if len(y) > 0 else np.nan

    blocks = [
        (["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat", "nr_bin", "attr_to_inst"], None,
         ["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat", "nr_bin", "attr_to_inst"]),
        (["freq_class"], ["mean", "sd", "min", "max"],
         ["freq_class_mean", "freq_class_sd", "freq_class_min", "freq_class_max"]),
        (["skewness", "kurtosis", "cor"], ["mean", "sd", "min", "max"],
         ["skewness_mean", "skewness_sd", "skewness_min", "skewness_max",
          "kurtosis_mean", "kurtosis_sd", "kurtosis_min", "kurtosis_max",
          "cor_mean", "cor_sd", "cor_min", "cor_max"]),
        (["nr_cor_attr", "nr_outliers"], None, ["nr_cor_attr", "nr_outliers"]),
        (["class_ent", "eq_num_attr", "ns_ratio"], None, ["class_ent", "eq_num_attr", "ns_ratio"]),
        (["attr_ent", "mut_inf", "joint_ent"], ["mean", "sd", "min", "max"],
         ["attr_ent_mean", "attr_ent_sd", "attr_ent_min", "attr_ent_max",
          "mut_inf_mean", "mut_inf_sd", "mut_inf_min", "mut_inf_max",
          "joint_ent_mean", "joint_ent_sd", "joint_ent_min", "joint_ent_max"]),
        (["class_conc"], ["mean", "sd", "min", "max"],
         ["class_conc_mean", "class_conc_sd", "class_conc_min", "class_conc_max"]),
        (["density", "c1", "c2", "n1", "t2"], None, ["density", "c1", "c2", "n1", "t2"]),
        (["best_node", "random_node", "linear_discr", "elite_nn", "naive_bayes", "one_nn", "worst_node"],
         ["mean"], ["best_node_mean", "random_node_mean", "linear_discr_mean",
                    "elite_nn_mean", "naive_bayes_mean", "one_nn_mean", "worst_node_mean"]),
        (["nodes", "leaves"], None, ["nodes", "leaves"]),
        (["tree_depth", "leaves_branch", "nodes_per_level", "var_importance"], ["mean", "max"],
         ["tree_depth_mean", "tree_depth_max", "leaves_branch_mean", "leaves_branch_max",
          "nodes_per_level_mean", "nodes_per_level_max", "var_importance_mean", "var_importance_max"]),
        (["leaves_per_class"], ["mean", "sd", "min", "max"],
         ["leaves_per_class_mean", "leaves_per_class_sd", "leaves_per_class_min", "leaves_per_class_max"]),
    ]

    for features, summary, keys in blocks:
        try:
            block = extract_mfe_block(X_df, y, features=features, summary=summary)
            for key in keys:
                meta[key] = block.get(key, np.nan)
        except Exception:
            for key in keys:
                meta.setdefault(key, np.nan)

    try:
        X_num = X_df.select_dtypes(include=[np.number]).copy()
        if X_num.shape[1] > 0:
            means = X_num.mean(axis=0, skipna=True).replace(0, np.nan)
            stds = X_num.std(axis=0, skipna=True)
            vc = (stds / means).replace([np.inf, -np.inf], np.nan).dropna()
            meta["var_coefficient_mean"] = to_float(vc.mean()) if len(vc) else np.nan
            meta["var_coefficient_sd"] = to_float(vc.std()) if len(vc) else np.nan
            meta["var_coefficient_min"] = to_float(vc.min()) if len(vc) else np.nan
            meta["var_coefficient_max"] = to_float(vc.max()) if len(vc) else np.nan
        else:
            for key in ["var_coefficient_mean", "var_coefficient_sd", "var_coefficient_min", "var_coefficient_max"]:
                meta[key] = np.nan
    except Exception:
        for key in ["var_coefficient_mean", "var_coefficient_sd", "var_coefficient_min", "var_coefficient_max"]:
            meta.setdefault(key, np.nan)

    try:
        block = extract_mfe_block(X_df, y, features=["f1", "n2", "n3", "n4"], summary=["mean"])
        meta["f1"] = block.get("f1_mean", np.nan)
        meta["n2_mean"] = block.get("n2_mean", np.nan)
        meta["n3_mean"] = block.get("n3_mean", np.nan)
        meta["n4"] = block.get("n4_mean", np.nan)
    except Exception:
        for key in ["f1", "n2_mean", "n3_mean", "n4"]:
            meta.setdefault(key, np.nan)

    return meta

