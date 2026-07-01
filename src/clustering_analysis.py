# =============================================================================
# EXPERIMENTO 2 — CELDA 3: Visualizaciones del clustering k=3
# =============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ─── Reproducir datos ────────────────────────────────────────────────────────
score_cols   = [f"score_{s}" for s in ["chi2", "mutual_info", "f_classif"]]
cols_excluir = ["dataset_name", "best_selector"] + score_cols

X_raw = meta_dataset_df.drop(columns=cols_excluir, errors='ignore')
X_raw = X_raw.dropna(axis=1, how="all")
X_raw = X_raw.loc[:, X_raw.isnull().mean() < 0.50]

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
X_sc    = pd.DataFrame(scaler.fit_transform(X_imp),  columns=X_raw.columns)

y_label = meta_dataset_df["best_selector"].values

# ─── Clustering k=3 ──────────────────────────────────────────────────────────
km     = KMeans(n_clusters=3, random_state=42, n_init=20)
labels = km.fit_predict(X_sc.values)

# ─── Proyección 2D para visualización ────────────────────────────────────────
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_sc.values)

# ─── Colores ─────────────────────────────────────────────────────────────────
colores_cluster = {0: "#C44E52", 1: "#55A868", 2: "#4C72B0"}
nombres_cluster = {
    0: "Cluster 0 — f_classif\n(confianza media)",
    1: "Cluster 1 — mutual_info\n(confianza alta)",
    2: "Cluster 2 — ambiguo\n(confianza baja)",
}
colores_selector = {
    "chi2":        "#DD8452",
    "mutual_info": "#55A868",
    "f_classif":   "#8172B3",
}

# =============================================================================
# GRÁFICA 1 — Estructura de clusters en el meta-dataset
# Cada punto es un dataset. Los colores muestran los 3 grupos encontrados.
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

for c, color in colores_cluster.items():
    mask = labels == c
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=color, label=nombres_cluster[c],
        alpha=0.75, s=60, edgecolors="white", linewidth=0.5
    )

ax.set_title(
    "Estructura de clusters en el meta-dataset\n"
    "Cada punto representa un dataset",
    fontweight="bold", fontsize=12
)
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("clustering_estructura.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# GRÁFICA 2 — Distribución de best_selector por cluster
# Muestra qué selector domina en cada grupo y con qué confianza
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

info_clusters = {
    0: ("Cluster 0 — f_classif", "Confianza MEDIA", "#C44E52"),
    1: ("Cluster 1 — mutual_info", "Confianza ALTA",  "#55A868"),
    2: ("Cluster 2 — ambiguo",     "Confianza BAJA",  "#4C72B0"),
}

df_an = meta_dataset_df.copy()
df_an["cluster"] = labels

for i, c in enumerate([0, 1, 2]):
    sub    = df_an[df_an["cluster"] == c]
    vc     = sub["best_selector"].value_counts()
    titulo, confianza, color_borde = info_clusters[c]

    bars = axes[i].bar(
        vc.index, vc.values,
        color=[colores_selector[s] for s in vc.index],
        edgecolor=color_borde, linewidth=2
    )
    axes[i].set_title(
        f"{titulo}\n({len(sub)} datasets) — {confianza}",
        fontweight="bold", fontsize=9
    )
    axes[i].set_ylim(0, 50)
    axes[i].grid(axis="y", alpha=0.3)

    # Número y porcentaje encima de cada barra
    for bar, (sel, cnt) in zip(bars, vc.items()):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 0.5, f"{cnt}\n({cnt/len(sub):.0%})",
            ha="center", fontsize=9, fontweight="bold"
        )

# Leyenda común
patches = [mpatches.Patch(color=c, label=s)
           for s, c in colores_selector.items()]
fig.legend(handles=patches, loc="upper right",
           fontsize=9, title="Selector")

plt.suptitle("Distribución de best_selector por cluster",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("clustering_selectores.png", dpi=150, bbox_inches="tight")
plt.show()

print("Gráficas guardadas:")
print("  clustering_estructura.png")
print("  clustering_selectores.png")

# =============================================================================
# EXPERIMENTO 2 — CELDA 4: Conexión clustering con modelo supervisado (GB)
# Pregunta: ¿acierta GB más en clusters con selector dominante claro?
# =============================================================================
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# ─── Generar predicciones GB Top 30 ──────────────────────────────────────────
# Se genera aquí directamente para no depender de variables de celdas anteriores
feats  = features_ordenadas[:30]
X_sel  = X_sc[feats].values

loo    = LeaveOneOut()
y_pred_gb_top30 = []

for train_idx, test_idx in loo.split(X_sel):
    X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
    y_tr       = y[train_idx]
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_tr, y_tr)
    y_pred_gb_top30.append(gb.predict(X_te)[0])

print(f"GB Top 30 — LOO Accuracy: {accuracy_score(y, y_pred_gb_top30):.4f}\n")

# ─── Análisis por cluster ─────────────────────────────────────────────────────
df_an = meta_dataset_df.copy()
df_an["cluster"]     = labels
df_an["pred_gb"]     = y_pred_gb_top30
df_an["correcto_gb"] = [r == p for r, p in
                        zip(df_an["best_selector"], df_an["pred_gb"])]

info_clusters = {
    0: ("Cluster 0 — f_classif",   "MEDIA"),
    1: ("Cluster 1 — mutual_info", "ALTA"),
    2: ("Cluster 2 — ambiguo",     "BAJA"),
}

print("=" * 55)
print("ACCURACY DE GB POR CLUSTER")
print("=" * 55)

for c in [0, 1, 2]:
    sub             = df_an[df_an["cluster"] == c]
    nombre, confianza = info_clusters[c]
    acc             = sub["correcto_gb"].mean()
    n               = len(sub)
    n_ok            = sub["correcto_gb"].sum()
    print(f"\n{nombre} (confianza {confianza}, {n} datasets)")
    print(f"  Aciertos: {n_ok}/{n} → accuracy = {acc:.4f}")



# =============================================================================
# EXPERIMENTO 2 — Caracterización de datasets por cluster
# ¿Qué tipo de datasets agrupa cada cluster?
# =============================================================================


# ─── Preparar dataframe ───────────────────────────────────────────────────────
# Usamos meta_dataset_df SIN escalar para que los valores sean interpretables
# (nr_inst en número real de instancias, class_ent en bits reales, etc.)
df_clusters = meta_dataset_df.copy()
df_clusters["cluster"] = labels

nombres_cluster = {
    0: "C0 — f_classif",
    1: "C1 — mutual_info",
    2: "C2 — ambiguo",
}
df_clusters["cluster_nombre"] = df_clusters["cluster"].map(nombres_cluster)

# ─── Estadísticas a analizar ─────────────────────────────────────────────────
estadisticas = {
    "nr_inst":             "Nº instancias",
    "nr_attr":             "Nº features",
    "nr_class":            "Nº clases",
    "class_ent":           "Entropía de clases (bits)",
    "majority_class_error":"Error clase mayoritaria",
    "class_balance_ratio": "Ratio de balance",
    "nr_num":              "Nº features numéricas",
    "nr_cat":              "Nº features categóricas",
}

# =============================================================================
# TABLA — Media y desviación de cada estadística por cluster
# =============================================================================
print("=" * 70)
print("CARACTERIZACIÓN DE DATASETS POR CLUSTER")
print("=" * 70)

filas = []
for feat, nombre in estadisticas.items():
    if feat not in df_clusters.columns:
        continue
    fila = {"Estadística": nombre}
    for c in [0, 1, 2]:
        sub = df_clusters[df_clusters["cluster"] == c][feat].dropna()
        # Formato: media ± std
        fila[nombres_cluster[c]] = f"{sub.mean():.2f} ± {sub.std():.2f}"
    filas.append(fila)

tabla = pd.DataFrame(filas).set_index("Estadística")
print(tabla.to_string())

# =============================================================================
# GRÁFICA — Boxplots: distribución de cada estadística por cluster
# Boxplot es mejor que barras porque muestra la dispersión real,
# no solo la media — importante con pocos datasets por cluster
# =============================================================================
features_plot = [
    ("nr_inst",              "Nº instancias"),
    ("nr_attr",              "Nº features"),
    ("class_ent",            "Entropía de clases (bits)"),
    ("majority_class_error", "Error clase mayoritaria"),
    ("class_balance_ratio",  "Ratio de balance"),
    ("nr_class",             "Nº clases"),
]
features_plot = [(f, n) for f, n in features_plot if f in df_clusters.columns]

colores = {
    "C0 — f_classif":  "#C44E52",
    "C1 — mutual_info":"#55A868",
    "C2 — ambiguo":    "#4C72B0",
}
orden = ["C0 — f_classif", "C1 — mutual_info", "C2 — ambiguo"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

for i, (feat, nombre) in enumerate(features_plot):
    ax = axes[i]
    sns.boxplot(
        data=df_clusters,
        x="cluster_nombre",
        y=feat,
        palette=colores,
        order=orden,
        width=0.5,
        ax=ax,
    )
    ax.set_title(nombre, fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(orden, rotation=12, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

plt.suptitle(
    "Caracterización de datasets por cluster\n"
    "¿Qué tipo de datasets agrupa cada grupo?",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("clustering_caracterizacion.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGráfica guardada como 'clustering_caracterizacion.png'")