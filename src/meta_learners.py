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

from .config import NOMBRES_SELECTORES, SEED
from .preprocessing import imputar_y_escalar, preparar_meta_features


def evaluar_clasificador_loo(
    meta_dataset_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[float, pd.DataFrame]:
    """Evaluate the direct RandomForest meta-classifier with Leave-One-Out."""
    X_meta, y_label, dataset_names = preparar_meta_features(meta_dataset_df)
    X_meta_arr = X_meta.values

    loo = LeaveOneOut()
    y_pred_clf, y_real_clf, names_clf = [], [], []

    for train_idx, test_idx in loo.split(X_meta_arr):
        X_tr, X_te = X_meta_arr[train_idx], X_meta_arr[test_idx]
        y_tr = y_label[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        clf = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf.fit(X_tr, y_tr)

        y_pred_clf.append(clf.predict(X_te)[0])
        y_real_clf.append(y_label[test_idx[0]])
        names_clf.append(dataset_names[test_idx[0]])

    loo_acc_clf = accuracy_score(y_real_clf, y_pred_clf)
    resultados_clf_df = pd.DataFrame({
        "dataset_name": names_clf,
        "real_selector": y_real_clf,
        "predicted_selector": y_pred_clf,
        "correcto": [r == p for r, p in zip(y_real_clf, y_pred_clf)],
    })
    return loo_acc_clf, resultados_clf_df


def evaluar_hibrido_loo(
    meta_dataset_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[float, list[str]]:
    """Evaluate the chi2-vs-rest classifier plus Ridge regressors."""
    score_cols = [f"score_{s}" for s in NOMBRES_SELECTORES]
    cols_excluir = ["dataset_name", "best_selector"] + score_cols

    X_df = meta_dataset_df.drop(columns=cols_excluir)
    X_df = X_df.dropna(axis=1, how="all")
    y = meta_dataset_df["best_selector"].values
    scores = meta_dataset_df[score_cols].values

    idx_f_classif = NOMBRES_SELECTORES.index("f_classif")
    idx_mutual_info = NOMBRES_SELECTORES.index("mutual_info")

    loo = LeaveOneOut()
    y_pred_hibrido = []

    for train_idx, test_idx in loo.split(X_df):
        X_tr = X_df.iloc[train_idx]
        X_te = X_df.iloc[test_idx]
        y_tr = y[train_idx]
        scores_tr = scores[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        y_A = np.where(y_tr == "chi2", "chi2", "no_chi2")
        clf_A = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf_A.fit(X_tr_imp, y_A)

        mask_no_chi2 = y_tr != "chi2"
        X_B = X_tr_imp[mask_no_chi2]
        targets_B = scores_tr[mask_no_chi2][:, [idx_f_classif, idx_mutual_info]]
        reg_B = Pipeline([
            ("scaler", StandardScaler()),
            ("multi_ridge", MultiOutputRegressor(Ridge(alpha=1.0))),
        ])
        reg_B.fit(X_B, targets_B)

        if clf_A.predict(X_te_imp)[0] == "chi2":
            y_pred_hibrido.append("chi2")
        else:
            pred_scores = reg_B.predict(X_te_imp)[0]
            if pred_scores[0] > pred_scores[1]:
                y_pred_hibrido.append("f_classif")
            else:
                y_pred_hibrido.append("mutual_info")

    return accuracy_score(y, y_pred_hibrido), y_pred_hibrido


def construir_ensemble(seed: int = SEED) -> VotingClassifier:
    """Build the soft-voting ensemble used in the notebook."""
    estimadores = [
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
    return VotingClassifier(estimators=estimadores, voting="soft")


def evaluar_ensemble_loo(
    meta_dataset_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    """Evaluate the soft-voting ensemble with Leave-One-Out."""
    X_meta, y_label, dataset_names = preparar_meta_features(meta_dataset_df)
    X_meta_arr = X_meta.values

    loo = LeaveOneOut()
    y_pred_ensemble, y_real, names_ens = [], [], []
    y_pred_individual = {nombre: [] for nombre in ["rf", "gb", "lr", "knn"]}

    for train_idx, test_idx in loo.split(X_meta_arr):
        X_tr, X_te = X_meta_arr[train_idx], X_meta_arr[test_idx]
        y_tr = y_label[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        ensemble = construir_ensemble(seed=seed)
        ensemble.fit(X_tr, y_tr)

        y_pred_ensemble.append(ensemble.predict(X_te)[0])
        y_real.append(y_label[test_idx[0]])
        names_ens.append(dataset_names[test_idx[0]])

        for nombre, modelo in ensemble.named_estimators_.items():
            pred_encoded = modelo.predict(X_te)[0]
            pred_decoded = ensemble.le_.inverse_transform([pred_encoded])[0]
            y_pred_individual[nombre].append(pred_decoded)

    acc_ensemble = accuracy_score(y_real, y_pred_ensemble)
    acc_individual = {
        nombre: accuracy_score(y_real, predicciones)
        for nombre, predicciones in y_pred_individual.items()
    }
    resultados_ensemble_df = pd.DataFrame({
        "dataset_name": names_ens,
        "real_selector": y_real,
        "predicted_selector": y_pred_ensemble,
        "correcto": [r == p for r, p in zip(y_real, y_pred_ensemble)],
    })
    return acc_ensemble, acc_individual, resultados_ensemble_df


def calcular_importancia_meta_features(
    meta_dataset_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Fit RF on scaled meta-features and return global importances."""
    X, y, _ = preparar_meta_features(meta_dataset_df, max_missing_ratio=0.50)
    X_sc = imputar_y_escalar(X)

    rf_analisis = RandomForestClassifier(n_estimators=500, random_state=seed)
    rf_analisis.fit(X_sc.values, y)

    importancias = pd.Series(
        rf_analisis.feature_importances_,
        index=X_sc.columns,
    ).sort_values(ascending=False)
    return importancias, X_sc, y


def evaluar_ablacion_features(
    X_sc: pd.DataFrame,
    y: np.ndarray,
    features_ordenadas: list[str],
    cortes: list[int] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate RF and GB with the top-k ranked meta-features."""
    cortes = [30, 40, 50, 60, 70, 80] if cortes is None else cortes
    loo = LeaveOneOut()
    filas = []

    for n_features in cortes:
        feats = features_ordenadas[:n_features]
        X_sel = X_sc[feats].values
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

        filas.append({
            "n_features": n_features,
            "rf_accuracy": accuracy_score(y, y_rf),
            "gb_accuracy": accuracy_score(y, y_gb),
        })

    return pd.DataFrame(filas)


def predecir_gb_top_k_loo(
    X_sc: pd.DataFrame,
    y: np.ndarray,
    features_ordenadas: list[str],
    k: int = 30,
    seed: int = SEED,
) -> list[str]:
    """Generate Leave-One-Out predictions for GB using the top-k features."""
    feats = features_ordenadas[:k]
    X_sel = X_sc[feats].values
    loo = LeaveOneOut()
    y_pred_gb_top = []

    for train_idx, test_idx in loo.split(X_sel):
        X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
        y_tr = y[train_idx]
        gb = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        gb.fit(X_tr, y_tr)
        y_pred_gb_top.append(gb.predict(X_te)[0])

    return y_pred_gb_top

