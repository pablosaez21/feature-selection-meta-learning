import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Recuperar meta-dataset guardado
meta_dataset_df = pd.read_pickle("meta_dataset_df2.pkl")
print(f"Meta-dataset cargado: {meta_dataset_df.shape[0]} datasets × {meta_dataset_df.shape[1]} columnas")

# Preparar X e y (base para todos los modelos)
score_cols   = ["score_chi2", "score_mutual_info", "score_f_classif"]
cols_excluir = ["dataset_name", "best_selector"] + score_cols
NOMBRES_SELECTORES = ["chi2", "mutual_info", "f_classif"]
SEED = 42

X_raw = meta_dataset_df.drop(columns=cols_excluir, errors='ignore')
X_raw = X_raw.dropna(axis=1, how="all")
X_raw = X_raw.loc[:, X_raw.isnull().mean() < 0.50]
y     = meta_dataset_df["best_selector"].values

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
X_sc    = pd.DataFrame(scaler.fit_transform(X_imp),  columns=X_raw.columns)

print(f"X lista: {X_sc.shape[0]} datasets × {X_sc.shape[1]} features")
print(f"Distribución y: {pd.Series(y).value_counts().to_dict()}")

from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICADOR LOO
# Target: best_selector (etiqueta discreta del selector ganador por score compuesto)
# Aprende directamente la clase, sin pasar por scores intermedios.
# ─────────────────────────────────────────────────────────────────────────────

# Columnas que NO son meta-features y por tanto NO entran en X
score_cols   = [f"score_{sel}" for sel in NOMBRES_SELECTORES]
cols_excluir = ["dataset_name", "best_selector"] + score_cols

# X: solo meta-features (input del modelo)
X_meta = meta_dataset_df.drop(columns=cols_excluir).copy()
X_meta = X_meta.dropna(axis=1, how="all")  # quitar columnas que sean 100% NaN

# y: etiqueta discreta = mejor selector por score compuesto
y_label = meta_dataset_df["best_selector"].values

dataset_names = meta_dataset_df["dataset_name"].copy()
X_meta_arr    = X_meta.values

loo = LeaveOneOut()
y_pred_clf, y_real_clf, names_clf = [], [], []

for train_idx, test_idx in loo.split(X_meta_arr):
    X_tr, X_te = X_meta_arr[train_idx], X_meta_arr[test_idx]
    y_tr       = y_label[train_idx]

    # Imputación DENTRO del fold para evitar data leakage:
    # la mediana se calcula solo con los datos de entrenamiento del fold
    imputer = SimpleImputer(strategy="median")
    X_tr    = imputer.fit_transform(X_tr)
    X_te    = imputer.transform(X_te)

    # Clasificador estándar: predice una de las 3 clases directamente
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf.fit(X_tr, y_tr)

    y_pred_clf.append(clf.predict(X_te)[0])
    y_real_clf.append(y_label[test_idx[0]])
    names_clf.append(dataset_names.iloc[test_idx[0]])

loo_acc_clf = accuracy_score(y_real_clf, y_pred_clf)
print(f"LOO Accuracy — CLASIFICADOR: {loo_acc_clf:.4f}")

resultados_clf_df = pd.DataFrame({
    "dataset_name":       names_clf,
    "real_selector":      y_real_clf,
    "predicted_selector": y_pred_clf,
    "correcto":           [r == p for r, p in zip(y_real_clf, y_pred_clf)],
})
resultados_clf_df



# =============================================================================
# CELDA: Enfoque híbrido — clasificación nivel A + regresión nivel B
# =============================================================================
# Nivel A (clasificador): chi2 vs no_chi2 → entrenado con TODO el train.
# Nivel B (regresor):     2 regresores Ridge que predicen score_f_classif
#                          y score_mutual_info, SOLO con muestras no_chi2.
#                          Decisión por argmax de los 2 scores predichos.
#
# Justificación: el nivel B tiene menos muestras (no_chi2). Con tan poca data
# un clasificador binario aprovecha mal la información, mientras que el score
# continuo da más señal. Aplicamos regresión justo donde la cadena flaquea.
# =============================================================================


# ─── 1) Preparar X (meta-features), y (target), scores (para el nivel B) ────
score_cols   = [f"score_{s}" for s in NOMBRES_SELECTORES]
cols_excluir = ["dataset_name", "best_selector"] + score_cols

X_df    = meta_dataset_df.drop(columns=cols_excluir)
X_df = X_df.dropna(axis=1, how="all")
y       = meta_dataset_df["best_selector"].values
scores  = meta_dataset_df[score_cols].values   # matriz (n_datasets, 3)

# Índices de los 2 selectores que entran al regresor B (chi2 queda fuera)
idx_f_classif   = NOMBRES_SELECTORES.index("f_classif")
idx_mutual_info = NOMBRES_SELECTORES.index("mutual_info")


# ─── 2) Evaluación LOO del híbrido ──────────────────────────────────────────
loo = LeaveOneOut()
y_pred_hibrido = []

for train_idx, test_idx in loo.split(X_df):
    X_tr      = X_df.iloc[train_idx]
    X_te      = X_df.iloc[test_idx]
    y_tr      = y[train_idx]
    scores_tr = scores[train_idx]

    # Imputación dentro del fold (la mediana sale solo del train)
    imputer  = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    # Nivel A: clasificador binario con TODO el train
    
    y_A   = np.where(y_tr == "chi2", "chi2", "no_chi2")#esto convierte las etiquetas en chi2 o no_chi2
    clf_A = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf_A.fit(X_tr_imp, y_A)

    # Nivel B: 2 regresores Ridge entrenados SOLO con no_chi2
    mask_no_chi2 = (y_tr != "chi2")
    X_B          = X_tr_imp[mask_no_chi2]
    targets_B    = scores_tr[mask_no_chi2][:, [idx_f_classif, idx_mutual_info]]
    reg_B = Pipeline([
        ("scaler",     StandardScaler()),
        ("multi_ridge", MultiOutputRegressor(Ridge(alpha=1.0))),
    ])
    reg_B.fit(X_B, targets_B)

    # Predicción top-down
    if clf_A.predict(X_te_imp)[0] == "chi2":
        y_pred_hibrido.append("chi2")
    else:
        # El regresor predice [score_f_classif, score_mutual_info] → argmax decide
        pred_scores = reg_B.predict(X_te_imp)[0]
        if pred_scores[0] > pred_scores[1]:
            y_pred_hibrido.append("f_classif")
        else:
            y_pred_hibrido.append("mutual_info")


# ─── 3) Resultado ───────────────────────────────────────────────────────────
acc_hibrido = accuracy_score(y, y_pred_hibrido)
print(f"Accuracy LOO (híbrido clasificación + regresión): {acc_hibrido:.4f}")

# =============================================================================
# MODELO 5 — ENSEMBLE SOFT VOTING (LOO)
# =============================================================================
# 4 modelos de paradigmas distintos votan por la clase más probable
# promediando sus predict_proba. Se evalúa con LOO igual que los demás.
# =============================================================================


# ─── 1) Preparar X e y (mismo formato que el clasificador estándar) ─────────
score_cols   = [f"score_{sel}" for sel in NOMBRES_SELECTORES]
cols_excluir = ["dataset_name", "best_selector"] + score_cols

X_meta = meta_dataset_df.drop(columns=cols_excluir).copy()
X_meta = X_meta.dropna(axis=1, how="all")  # quitar columnas 100% NaN

y_label       = meta_dataset_df["best_selector"].values
dataset_names = meta_dataset_df["dataset_name"].copy()
X_meta_arr    = X_meta.values


# ─── 2) Función que construye el ensemble desde cero en cada fold ───────────

def construir_ensemble(seed):
    estimadores = [
        # RandomForest: ensemble de árboles, no necesita scaler
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=seed
        )),
        # GradientBoosting: boosting, sesgo distinto al RF, tampoco necesita scaler
        ("gb", GradientBoostingClassifier(
            n_estimators=150, random_state=seed
        )),
        # LogisticRegression: modelo lineal → SÍ necesita scaler
        # max_iter=2000 para asegurar convergencia con 3 clases y pocos datos
        ("lr", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, random_state=seed)),
        ])),
        # KNN: instance-based, basado en distancias → SÍ necesita scaler
        # k=10 es razonable con ~107 muestras y 3 clases
        ("knn", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=10)),
        ])),
    ]
    # voting='soft' → promedia predict_proba de los 4 modelos y decide por argmax
    return VotingClassifier(estimators=estimadores, voting="soft")


# ─── 3) Evaluación LOO ──────────────────────────────────────────────────────
loo = LeaveOneOut()

# Listas para predicciones del ensemble Y de cada modelo individual
# (las individuales sirven para analizar si los errores están correlacionados)
y_pred_ensemble, y_real, names_ens = [], [], []
y_pred_individual = {nombre: [] for nombre in ["rf", "gb", "lr", "knn"]}

for train_idx, test_idx in loo.split(X_meta_arr):
    X_tr, X_te = X_meta_arr[train_idx], X_meta_arr[test_idx]
    y_tr       = y_label[train_idx]

    # Imputación dentro del fold: 
    imputer = SimpleImputer(strategy="median")
    X_tr    = imputer.fit_transform(X_tr)
    X_te    = imputer.transform(X_te)

    # Construir y entrenar el ensemble (entrena los 4 modelos por debajo)
    ensemble = construir_ensemble(seed=SEED)
    ensemble.fit(X_tr, y_tr)

    # Predicción del ensemble (ya combina las 4 probabilidades por dentro)
    y_pred_ensemble.append(ensemble.predict(X_te)[0])
    y_real.append(y_label[test_idx[0]])
    names_ens.append(dataset_names.iloc[test_idx[0]])

    # Predicciones individuales: decodificamos con el LabelEncoder
    # interno del VotingClassifier (ensemble.le_) para recuperar
    # los strings originales ("chi2", "mutual_info", "f_classif")
    for nombre, modelo in ensemble.named_estimators_.items():
        pred_encoded = modelo.predict(X_te)[0]   # devuelve 0, 1 o 2
        pred_decoded = ensemble.le_.inverse_transform([pred_encoded])[0]  # → string
        y_pred_individual[nombre].append(pred_decoded)


# ─── 4) Accuracy del ensemble y de cada modelo individual ───────────────────
loo_acc_ensemble = accuracy_score(y_real, y_pred_ensemble)
print(f"LOO Accuracy — ENSEMBLE soft voting: {loo_acc_ensemble:.4f}")
print()
print("Accuracy individual de cada modelo del ensemble:")
for nombre, predicciones in y_pred_individual.items():
    acc_ind = accuracy_score(y_real, predicciones)
    print(f"  {nombre:<5} → {acc_ind:.4f}")


# ─── 6) DataFrame final con resultados ──────────────────────────────────────
resultados_ensemble_df = pd.DataFrame({
    "dataset_name":       names_ens,
    "real_selector":      y_real,
    "predicted_selector": y_pred_ensemble,
    "correcto":           [r == p for r, p in zip(y_real, y_pred_ensemble)],
})
resultados_ensemble_df

# =============================================================================
# ANÁLISIS DE META-FEATURES 
# Orden: escalar → importancia RF 
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ─── Preparar X e y ──────────────────────────────────────────────────────────
score_cols   = [f"score_{sel}" for sel in ["chi2", "mutual_info", "f_classif"]]
cols_excluir = ["dataset_name", "best_selector"] + score_cols

X = meta_dataset_df.drop(columns=cols_excluir, errors='ignore')
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.isnull().mean() < 0.50]
y = meta_dataset_df["best_selector"].values

print(f"Features iniciales: {X.shape[1]}")
print(f"Datasets: {X.shape[0]}\n")

# Imputar y escalar ANTES de cualquier análisis
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_sc    = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns)

# =============================================================================
# IMPORTANCIA SEGÚN RANDOMFOREST
# RF entrenado UNA VEZ sobre todos los datos → importancia global estable.
# No nos importa el accuracy, solo qué features usa para distinguir selectores.
# Usamos las 80 features completas — en análisis exploratorio más info es mejor.
# =============================================================================
rf_analisis = RandomForestClassifier(n_estimators=500, random_state=42) #aumento el número de árboles para calcular la importancia de forma más estable
rf_analisis.fit(X_sc.values, y)

importancias = pd.Series(
    rf_analisis.feature_importances_,
    index=X_sc.columns
).sort_values(ascending=False)

print(f"{'Feature':<30} {'Importancia':>12} {'Acumulada':>10}")
print("-" * 55)
acumulada = 0
for feat, imp in importancias.items():
    acumulada += imp
    print(f"{feat:<30} {imp:>12.4f} {acumulada:>9.1%}")

plt.figure(figsize=(10, 8))
importancias.head(25).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 25 meta-features por importancia (RF)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

importancia_acumulada = importancias.cumsum()
for pct in [0.80, 0.90, 0.95]:
    n = (importancia_acumulada < pct).sum() + 1
    print(f"Features para {pct:.0%} importancia: {n}")

features_ordenadas = importancias.index.tolist()

# =============================================================================
# ABLACIÓN DE FEATURES — RF y GB con distintos cortes de features
# =============================================================================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# features_ordenadas viene de la celda anterior (ranking sobre 80 features)
loo = LeaveOneOut()

print(f"{'Configuración':<20} {'RF':>8} {'GB':>8}")
print("-" * 38)

for n_features in [30, 40, 50, 60, 70, 80]:
    feats = features_ordenadas[:n_features] #seleccionamos el número de features deseado, por orden de importancia
    X_sel = X_sc[feats].values  # X_sc tiene las 80 features escaladas
    y_rf, y_gb = [], []

    for train_idx, test_idx in loo.split(X_sel):
        X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
        y_tr = y[train_idx]

        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

        rf.fit(X_tr, y_tr)
        gb.fit(X_tr, y_tr)

        y_rf.append(rf.predict(X_te)[0])
        y_gb.append(gb.predict(X_te)[0])

    acc_rf = accuracy_score(y, y_rf)
    acc_gb = accuracy_score(y, y_gb)
    print(f"Top {n_features:<16} {acc_rf:>8.4f} {acc_gb:>8.4f}")