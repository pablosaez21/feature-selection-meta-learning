"""Meta-learner evaluation and feature-importance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED, SELECTOR_NAMES
from .preprocessing import impute_and_scale, split_meta_dataset


def evaluate_direct_classifier_loo(meta_dataset_df: pd.DataFrame, seed: int = SEED):
    """Evaluate the direct RandomForest meta-classifier with Leave-One-Out."""
    X_meta, y_label, dataset_names = split_meta_dataset(meta_dataset_df)
    X_meta_arr = X_meta.values
    loo = LeaveOneOut()
    y_pred, y_real, names = [], [], []

    for train_idx, test_idx in loo.split(X_meta_arr):
        X_tr, X_te = X_meta_arr[train_idx], X_meta_arr[test_idx]
        y_tr = y_label[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        clf = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf.fit(X_tr, y_tr)

        y_pred.append(clf.predict(X_te)[0])
        y_real.append(y_label[test_idx[0]])
        names.append(dataset_names[test_idx[0]])

    results_df = pd.DataFrame({
        "dataset_name": names,
        "real_selector": y_real,
        "predicted_selector": y_pred,
        "correct": [r == p for r, p in zip(y_real, y_pred)],
    })
    return accuracy_score(y_real, y_pred), results_df


def evaluate_hybrid_loo(meta_dataset_df: pd.DataFrame, seed: int = SEED):
    """Evaluate the chi2-vs-rest classifier plus Ridge score regressors."""
    score_cols = [f"score_{selector}" for selector in SELECTOR_NAMES]
    excluded_cols = ["dataset_name", "best_selector"] + score_cols

    X_df = meta_dataset_df.drop(columns=excluded_cols)
    X_df = X_df.dropna(axis=1, how="all")
    y = meta_dataset_df["best_selector"].values
    scores = meta_dataset_df[score_cols].values

    idx_f_classif = SELECTOR_NAMES.index("f_classif")
    idx_mutual_info = SELECTOR_NAMES.index("mutual_info")

    loo = LeaveOneOut()
    y_pred = []

    for train_idx, test_idx in loo.split(X_df):
        X_tr = X_df.iloc[train_idx]
        X_te = X_df.iloc[test_idx]
        y_tr = y[train_idx]
        scores_tr = scores[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        y_level_a = np.where(y_tr == "chi2", "chi2", "no_chi2")
        clf_a = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf_a.fit(X_tr_imp, y_level_a)

        mask_no_chi2 = y_tr != "chi2"
        X_b = X_tr_imp[mask_no_chi2]
        targets_b = scores_tr[mask_no_chi2][:, [idx_f_classif, idx_mutual_info]]
        reg_b = Pipeline([
            ("scaler", StandardScaler()),
            ("multi_ridge", MultiOutputRegressor(Ridge(alpha=1.0))),
        ])
        reg_b.fit(X_b, targets_b)

        if clf_a.predict(X_te_imp)[0] == "chi2":
            y_pred.append("chi2")
        else:
            pred_scores = reg_b.predict(X_te_imp)[0]
            y_pred.append("f_classif" if pred_scores[0] > pred_scores[1] else "mutual_info")

    return accuracy_score(y, y_pred), y_pred


def build_soft_voting_ensemble(seed: int = SEED) -> VotingClassifier:
    """Build the soft-voting ensemble used in the notebook."""
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=seed)),
        ("gb", GradientBoostingClassifier(n_estimators=150, random_state=seed)),
        ("lr", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ])),
        ("knn", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=10)),
        ])),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


def evaluate_ensemble_loo(meta_dataset_df: pd.DataFrame, seed: int = SEED):
    """Evaluate the soft-voting ensemble with Leave-One-Out."""
    X_meta, y_label, dataset_names = split_meta_dataset(meta_dataset_df)
    X_meta_arr = X_meta.values
    loo = LeaveOneOut()
    y_pred_ensemble, y_real, names = [], [], []
    individual_preds = {name: [] for name in ["rf", "gb", "lr", "knn"]}

    for train_idx, test_idx in loo.split(X_meta_arr):
        X_tr, X_te = X_meta_arr[train_idx], X_meta_arr[test_idx]
        y_tr = y_label[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        ensemble = build_soft_voting_ensemble(seed=seed)
        ensemble.fit(X_tr, y_tr)

        y_pred_ensemble.append(ensemble.predict(X_te)[0])
        y_real.append(y_label[test_idx[0]])
        names.append(dataset_names[test_idx[0]])

        for name, model in ensemble.named_estimators_.items():
            pred_encoded = model.predict(X_te)[0]
            pred_decoded = ensemble.le_.inverse_transform([pred_encoded])[0]
            individual_preds[name].append(pred_decoded)

    individual_accuracy = {
        name: accuracy_score(y_real, preds)
        for name, preds in individual_preds.items()
    }
    results_df = pd.DataFrame({
        "dataset_name": names,
        "real_selector": y_real,
        "predicted_selector": y_pred_ensemble,
        "correct": [r == p for r, p in zip(y_real, y_pred_ensemble)],
    })
    return accuracy_score(y_real, y_pred_ensemble), individual_accuracy, results_df


def compute_meta_feature_importance(meta_dataset_df: pd.DataFrame, seed: int = SEED):
    """Train RF on scaled meta-features and return global importances."""
    X, y, _ = split_meta_dataset(meta_dataset_df, max_missing_ratio=0.50)
    X_scaled = impute_and_scale(X)

    rf = RandomForestClassifier(n_estimators=500, random_state=seed)
    rf.fit(X_scaled.values, y)

    importances = pd.Series(rf.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
    return importances, X_scaled, y


def evaluate_feature_ablation(
    X_scaled: pd.DataFrame,
    y: np.ndarray,
    ordered_features: list[str],
    cutoffs: list[int] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate RF and GB with different top-k feature cutoffs."""
    cutoffs = [30, 40, 50, 60, 70, 80] if cutoffs is None else cutoffs
    rows = []
    loo = LeaveOneOut()

    for n_features in cutoffs:
        features = ordered_features[:n_features]
        X_sel = X_scaled[features].values
        y_rf, y_gb = [], []

        for train_idx, test_idx in loo.split(X_sel):
            X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
            y_tr = y[train_idx]

            rf = RandomForestClassifier(n_estimators=200, random_state=seed)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_tr, y_tr)
            gb.fit(X_tr, y_tr)

            y_rf.append(rf.predict(X_te)[0])
            y_gb.append(gb.predict(X_te)[0])

        rows.append({
            "n_features": n_features,
            "random_forest_accuracy": accuracy_score(y, y_rf),
            "gradient_boosting_accuracy": accuracy_score(y, y_gb),
        })

    return pd.DataFrame(rows)


def predict_gb_top_k_loo(
    X_scaled: pd.DataFrame,
    y: np.ndarray,
    ordered_features: list[str],
    k: int = 30,
    seed: int = SEED,
) -> list[str]:
    """Generate Leave-One-Out predictions for GB using the top-k features."""
    X_sel = X_scaled[ordered_features[:k]].values
    loo = LeaveOneOut()
    y_pred = []

    for train_idx, test_idx in loo.split(X_sel):
        X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
        y_tr = y[train_idx]
        gb = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        gb.fit(X_tr, y_tr)
        y_pred.append(gb.predict(X_te)[0])

    return y_pred

