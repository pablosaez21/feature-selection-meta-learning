# =============================================================================
# GRÁFICA — Curva de apredndizaje de features
# =============================================================================
import matplotlib.pyplot as plt

n_features_list = [30, 40, 50, 60, 70, 80]
acc_rf = [0.5635, 0.5238, 0.5159, 0.5238, 0.5317, 0.4921]
acc_gb = [0.5873, 0.5397, 0.5317, 0.5000, 0.5397, 0.5317]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(n_features_list, acc_rf, "o-", color="#4C72B0",
        linewidth=2, markersize=7, label="RandomForest")
ax.plot(n_features_list, acc_gb, "s-", color="#C44E52",
        linewidth=2, markersize=7, label="GradientBoosting")

# Línea de referencia: baseline mayoría
ax.axhline(y=0.4127, color="gray", linestyle="--",
           linewidth=1.2, alpha=0.7, label="Baseline mayoría (0.4127)")

# Marcar el punto óptimo
ax.axvline(x=30, color="black", linestyle=":",
           linewidth=1.2, alpha=0.6)
ax.text(31, 0.515, "← Óptimo\n   (Top 30)",
        fontsize=9, color="black")

# Anotar los valores del Top 30
ax.annotate(f"0.5873", xy=(30, 0.5873), xytext=(33, 0.583),
            fontsize=9, color="#C44E52", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
ax.annotate(f"0.5635", xy=(30, 0.5635), xytext=(33, 0.557),
            fontsize=9, color="#4C72B0", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2))

ax.set_xlabel("Número de features (Top-k por importancia RF)", fontsize=11)
ax.set_ylabel("LOO Accuracy", fontsize=11)
ax.set_title("Curva de aprendizaje — impacto del número de meta-features\n"
             "en RF y GradientBoosting", fontsize=12, fontweight="bold")
ax.set_xticks(n_features_list)
ax.set_ylim(0.38, 0.62)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle=":")

plt.tight_layout()
plt.savefig("ablacion_features.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# GRÁFICA — Importancia de meta-features coloreada por grupo pymfe
# =============================================================================

# ─── Asignar cada feature a su grupo pymfe ───────────────────────────────────
grupos = {
    # General
    "nr_inst":              "General",
    "nr_attr":              "General",
    "nr_class":             "General",
    "nr_num":               "General",
    "nr_cat":               "General",
    "nr_bin":               "General",
    "attr_to_inst":         "General",
    "nr_outliers":          "General",
    # Statistical
    "skewness_mean":        "Statistical",
    "skewness_sd":          "Statistical",
    "skewness_min":         "Statistical",
    "skewness_max":         "Statistical",
    "kurtosis_mean":        "Statistical",
    "kurtosis_sd":          "Statistical",
    "kurtosis_max":         "Statistical",
    "cor_mean":             "Statistical",
    "cor_sd":               "Statistical",
    "cor_min":              "Statistical",
    "cor_max":              "Statistical",
    "var_coefficient_mean": "Statistical",
    "class_balance_ratio":  "Statistical",
    "freq_class_max":       "Statistical",
    "kurtosis_min":         "Statistical",
    "var_coefficient_sd":   "Statistical",
    "var_coefficient_min":  "Statistical",
    "var_coefficient_max":  "Statistical",
    "nr_cor_attr":          "Statistical",
    "majority_class_error": "Statistical",
    "freq_class_mean":      "Statistical",
    "freq_class_sd":        "Statistical",
    "freq_class_min":       "Statistical",
    "freq_class_max":       "Statistical",
    # Info-theory
    "class_ent":            "Info-theory",
    "attr_ent_mean":        "Info-theory",
    "attr_ent_sd":          "Info-theory",
    "attr_ent_min":         "Info-theory",
    "attr_ent_max":         "Info-theory",
    "mut_inf_mean":         "Info-theory",
    "mut_inf_sd":           "Info-theory",
    "mut_inf_min":          "Info-theory",
    "mut_inf_max":          "Info-theory",
    "joint_ent_mean":       "Info-theory",
    "joint_ent_min":        "Info-theory",
    "joint_ent_max":        "Info-theory",
    "class_conc_mean":      "Info-theory",
    "class_conc_sd":        "Info-theory",
    "class_conc_min":       "Info-theory",
    "class_conc_max":       "Info-theory",
    "eq_num_attr":          "Info-theory",
    "ns_ratio":             "Info-theory",
    "joint_ent_sd":         "Info-theory",
    "c2":                   "Complexity",
    # Complexity
    "density":              "Complexity",
    "c1":                   "Complexity",
    "n1":                   "Complexity",
    "t2":                   "Complexity",
    "f1":                   "Complexity",
    "n2_mean":              "Complexity",
    "n3_mean":              "Complexity",
    "n4":                   "Complexity",
    # Landmarking
    "best_node_mean":       "Landmarking",
    "random_node_mean":     "Landmarking",
    "linear_discr_mean":    "Landmarking",
    "elite_nn_mean":        "Landmarking",
    "naive_bayes_mean":     "Landmarking",
    "one_nn_mean":          "Landmarking",
    "worst_node_mean":      "Landmarking",
    # Model-based
    "nodes":                "Model-based",
    "tree_depth_mean":      "Model-based",
    "var_importance_mean":  "Model-based",
    "var_importance_max":   "Model-based",
    "leaves_per_class_sd":  "Model-based",
    "leaves_per_class_max": "Model-based",
    "leaves":               "Model-based",
    "tree_depth_max":       "Model-based",
    "leaves_branch_mean":   "Model-based",
    "leaves_branch_max":    "Model-based",
    "nodes_per_level_mean": "Model-based",
    "nodes_per_level_max":  "Model-based",
    "leaves_per_class_mean":"Model-based",
    "leaves_per_class_min": "Model-based",
    
}

# Colores por grupo
colores_grupo = {
    "General":     "#4C72B0",
    "Statistical":  "#DD8452",
    "Info-theory":  "#55A868",
    "Complexity":   "#C44E52",
    "Landmarking":  "#8172B3",
    "Model-based":  "#937860",
}

# Preparar datos del ranking (top 61 features)
n_mostrar = 80
feats_plot   = importancias.index[:n_mostrar].tolist()
vals_plot    = importancias.values[:n_mostrar]
colores_plot = [colores_grupo.get(grupos.get(f, "General"), "#999999")
                for f in feats_plot]

# ─── Figura principal ────────────────────────────────────────────────────────
fig, (ax_main, ax_pie) = plt.subplots(
    1, 2,
    figsize=(16, 14),
    gridspec_kw={"width_ratios": [3, 1]}
)

# ── Gráfica principal: barras horizontales ────────────────────────────────────
bars = ax_main.barh(
    range(n_mostrar), vals_plot,
    color=colores_plot, edgecolor="white", linewidth=0.5
)
ax_main.invert_yaxis()
ax_main.set_yticks(range(n_mostrar))
ax_main.set_yticklabels(feats_plot, fontsize=8.5)

# Línea de corte en top 30
ax_main.axhline(y=29.5, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
ax_main.text(
    vals_plot[0] * 0.6, 29.5 - 0.8,
    "← Top 30 seleccionadas",
    fontsize=9, color="black", fontweight="bold"
)

# Porcentaje individual de cada feature sobre el total
for i, (feat, val) in enumerate(zip(feats_plot, vals_plot)):
    ax_main.text(
        val + 0.0003, i,
        f"{val:.1%}",
        va="center", fontsize=7, color="#333333"
    )

ax_main.set_xlabel("Importancia (RF)", fontsize=11)
ax_main.set_title(
    "Importancia de meta-features por grupo pymfe\n"
    "(coloreado por grupo — línea indica corte Top 30)",
    fontsize=12, fontweight="bold", pad=12
)
ax_main.set_xlim(0, vals_plot[0] * 1.35)
ax_main.grid(axis="x", alpha=0.3, linestyle=":")

# Leyenda
patches = [
    mpatches.Patch(color=c, label=g)
    for g, c in colores_grupo.items()
]
ax_main.legend(
    handles=patches, loc="lower right",
    fontsize=9, title="Grupo pymfe", title_fontsize=9,
    framealpha=0.9
)

# ── Gráfico secundario: distribución por grupo en Top 30 ─────────────────────
top30_feats  = importancias.index[:30].tolist()
top30_grupos = [grupos.get(f, "General") for f in top30_feats]

conteo = {}
for g in top30_grupos:
    conteo[g] = conteo.get(g, 0) + 1

# Ordenar por conteo
conteo_sorted = dict(sorted(conteo.items(), key=lambda x: x[1], reverse=True))

ax_pie.barh(
    list(conteo_sorted.keys()),
    list(conteo_sorted.values()),
    color=[colores_grupo[g] for g in conteo_sorted.keys()],
    edgecolor="white"
)
ax_pie.set_xlabel("Nº features en Top 30", fontsize=10)
ax_pie.set_title("Distribución\npor grupo", fontsize=11, fontweight="bold")
ax_pie.grid(axis="x", alpha=0.3, linestyle=":")

for i, (g, n) in enumerate(conteo_sorted.items()):
    ax_pie.text(n + 0.05, i, str(n), va="center", fontsize=10, fontweight="bold")

plt.tight_layout(pad=2.0)
plt.savefig("importancia_meta_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfica guardada como 'importancia_meta_features.png'")