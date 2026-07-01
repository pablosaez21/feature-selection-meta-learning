"""Visualization helpers for the notebook analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURE_GROUPS = {
    "nr_inst": "General",
    "nr_attr": "General",
    "nr_class": "General",
    "nr_num": "General",
    "nr_cat": "General",
    "nr_bin": "General",
    "attr_to_inst": "General",
    "nr_outliers": "General",
    "skewness_mean": "Statistical",
    "skewness_sd": "Statistical",
    "skewness_min": "Statistical",
    "skewness_max": "Statistical",
    "kurtosis_mean": "Statistical",
    "kurtosis_sd": "Statistical",
    "kurtosis_min": "Statistical",
    "kurtosis_max": "Statistical",
    "cor_mean": "Statistical",
    "cor_sd": "Statistical",
    "cor_min": "Statistical",
    "cor_max": "Statistical",
    "var_coefficient_mean": "Statistical",
    "var_coefficient_sd": "Statistical",
    "var_coefficient_min": "Statistical",
    "var_coefficient_max": "Statistical",
    "class_balance_ratio": "Statistical",
    "majority_class_error": "Statistical",
    "freq_class_mean": "Statistical",
    "freq_class_sd": "Statistical",
    "freq_class_min": "Statistical",
    "freq_class_max": "Statistical",
    "nr_cor_attr": "Statistical",
    "class_ent": "Info-theory",
    "attr_ent_mean": "Info-theory",
    "attr_ent_sd": "Info-theory",
    "attr_ent_min": "Info-theory",
    "attr_ent_max": "Info-theory",
    "mut_inf_mean": "Info-theory",
    "mut_inf_sd": "Info-theory",
    "mut_inf_min": "Info-theory",
    "mut_inf_max": "Info-theory",
    "joint_ent_mean": "Info-theory",
    "joint_ent_sd": "Info-theory",
    "joint_ent_min": "Info-theory",
    "joint_ent_max": "Info-theory",
    "class_conc_mean": "Info-theory",
    "class_conc_sd": "Info-theory",
    "class_conc_min": "Info-theory",
    "class_conc_max": "Info-theory",
    "eq_num_attr": "Info-theory",
    "ns_ratio": "Info-theory",
    "density": "Complexity",
    "c1": "Complexity",
    "c2": "Complexity",
    "n1": "Complexity",
    "t2": "Complexity",
    "f1": "Complexity",
    "n2_mean": "Complexity",
    "n3_mean": "Complexity",
    "n4": "Complexity",
    "best_node_mean": "Landmarking",
    "random_node_mean": "Landmarking",
    "linear_discr_mean": "Landmarking",
    "elite_nn_mean": "Landmarking",
    "naive_bayes_mean": "Landmarking",
    "one_nn_mean": "Landmarking",
    "worst_node_mean": "Landmarking",
    "nodes": "Model-based",
    "leaves": "Model-based",
    "tree_depth_mean": "Model-based",
    "tree_depth_max": "Model-based",
    "leaves_branch_mean": "Model-based",
    "leaves_branch_max": "Model-based",
    "nodes_per_level_mean": "Model-based",
    "nodes_per_level_max": "Model-based",
    "var_importance_mean": "Model-based",
    "var_importance_max": "Model-based",
    "leaves_per_class_mean": "Model-based",
    "leaves_per_class_sd": "Model-based",
    "leaves_per_class_min": "Model-based",
    "leaves_per_class_max": "Model-based",
}

GROUP_COLORS = {
    "General": "#4C72B0",
    "Statistical": "#DD8452",
    "Info-theory": "#55A868",
    "Complexity": "#C44E52",
    "Landmarking": "#8172B3",
    "Model-based": "#937860",
}

CLUSTER_COLORS = {0: "#C44E52", 1: "#55A868", 2: "#4C72B0"}
CLUSTER_NAMES = {
    0: "Cluster 0 - f_classif\n(confianza media)",
    1: "Cluster 1 - mutual_info\n(confianza alta)",
    2: "Cluster 2 - ambiguo\n(confianza baja)",
}
SELECTOR_COLORS = {
    "chi2": "#DD8452",
    "mutual_info": "#55A868",
    "f_classif": "#8172B3",
}


def plot_top_importancias(importancias: pd.Series, top_n: int = 25) -> plt.Figure:
    """Plot the top RF meta-feature importances."""
    fig, ax = plt.subplots(figsize=(10, 8))
    importancias.head(top_n).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top 25 meta-features por importancia (RF)")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    return fig


def plot_curva_aprendizaje_features(
    resultados_ablacion: pd.DataFrame,
    baseline: float = 0.4127,
    output_path: str | Path | None = "figures/ablacion_features.png",
) -> plt.Figure:
    """Plot RF and GB accuracy by number of selected meta-features."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(resultados_ablacion["n_features"], resultados_ablacion["rf_accuracy"],
            "o-", color="#4C72B0", linewidth=2, markersize=7, label="RandomForest")
    ax.plot(resultados_ablacion["n_features"], resultados_ablacion["gb_accuracy"],
            "s-", color="#C44E52", linewidth=2, markersize=7, label="GradientBoosting")
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Baseline mayoria ({baseline:.4f})")
    ax.axvline(x=30, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Numero de features (Top-k por importancia RF)", fontsize=11)
    ax.set_ylabel("LOO Accuracy", fontsize=11)
    ax.set_title("Curva de aprendizaje - impacto del numero de meta-features\nen RF y GradientBoosting",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(resultados_ablacion["n_features"])
    ax.set_ylim(0.38, 0.62)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_importancia_por_grupo(
    importancias: pd.Series,
    n_mostrar: int = 80,
    output_path: str | Path | None = "figures/importancia_meta_features.png",
) -> plt.Figure:
    """Plot RF importances colored by PyMFE group."""
    feats_plot = importancias.index[:n_mostrar].tolist()
    vals_plot = importancias.values[:n_mostrar]
    colores_plot = [GROUP_COLORS.get(FEATURE_GROUPS.get(f, "General"), "#999999") for f in feats_plot]

    fig, (ax_main, ax_pie) = plt.subplots(
        1, 2, figsize=(16, 14), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax_main.barh(range(n_mostrar), vals_plot, color=colores_plot, edgecolor="white", linewidth=0.5)
    ax_main.invert_yaxis()
    ax_main.set_yticks(range(n_mostrar))
    ax_main.set_yticklabels(feats_plot, fontsize=8.5)
    ax_main.axhline(y=29.5, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_main.text(vals_plot[0] * 0.6, 28.7, "<- Top 30 seleccionadas",
                 fontsize=9, color="black", fontweight="bold")

    for i, val in enumerate(vals_plot):
        ax_main.text(val + 0.0003, i, f"{val:.1%}", va="center", fontsize=7, color="#333333")

    ax_main.set_xlabel("Importancia (RF)", fontsize=11)
    ax_main.set_title("Importancia de meta-features por grupo pymfe\n(coloreado por grupo - linea indica corte Top 30)",
                      fontsize=12, fontweight="bold", pad=12)
    ax_main.set_xlim(0, vals_plot[0] * 1.35)
    ax_main.grid(axis="x", alpha=0.3, linestyle=":")
    ax_main.legend(
        handles=[mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()],
        loc="lower right", fontsize=9, title="Grupo pymfe", title_fontsize=9, framealpha=0.9,
    )

    conteo = {}
    for feat in importancias.index[:30].tolist():
        grupo = FEATURE_GROUPS.get(feat, "General")
        conteo[grupo] = conteo.get(grupo, 0) + 1
    conteo_sorted = dict(sorted(conteo.items(), key=lambda x: x[1], reverse=True))

    ax_pie.barh(
        list(conteo_sorted.keys()),
        list(conteo_sorted.values()),
        color=[GROUP_COLORS[g] for g in conteo_sorted.keys()],
        edgecolor="white",
    )
    ax_pie.set_xlabel("N features en Top 30", fontsize=10)
    ax_pie.set_title("Distribucion\npor grupo", fontsize=11, fontweight="bold")
    ax_pie.grid(axis="x", alpha=0.3, linestyle=":")
    for i, (_, n) in enumerate(conteo_sorted.items()):
        ax_pie.text(n + 0.05, i, str(n), va="center", fontsize=10, fontweight="bold")

    fig.tight_layout(pad=2.0)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_clustering(
    meta_dataset_df: pd.DataFrame,
    labels,
    X_2d,
    output_dir: str | Path = "figures",
) -> tuple[plt.Figure, plt.Figure]:
    """Plot PCA cluster structure and selector distribution by cluster."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax = plt.subplots(figsize=(8, 6))
    for c, color in CLUSTER_COLORS.items():
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=CLUSTER_NAMES[c],
                   alpha=0.75, s=60, edgecolors="white", linewidth=0.5)
    ax.set_title("Estructura de clusters en el meta-dataset\nCada punto representa un dataset",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "clustering_estructura.png", dpi=150, bbox_inches="tight")

    fig2, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    info_clusters = {
        0: ("Cluster 0 - f_classif", "Confianza MEDIA", "#C44E52"),
        1: ("Cluster 1 - mutual_info", "Confianza ALTA", "#55A868"),
        2: ("Cluster 2 - ambiguo", "Confianza BAJA", "#4C72B0"),
    }
    df_an = meta_dataset_df.copy()
    df_an["cluster"] = labels

    for i, c in enumerate([0, 1, 2]):
        sub = df_an[df_an["cluster"] == c]
        vc = sub["best_selector"].value_counts()
        titulo, confianza, color_borde = info_clusters[c]
        bars = axes[i].bar(
            vc.index,
            vc.values,
            color=[SELECTOR_COLORS[s] for s in vc.index],
            edgecolor=color_borde,
            linewidth=2,
        )
        axes[i].set_title(f"{titulo}\n({len(sub)} datasets) - {confianza}",
                          fontweight="bold", fontsize=9)
        axes[i].set_ylim(0, 50)
        axes[i].grid(axis="y", alpha=0.3)
        for bar, (_, cnt) in zip(bars, vc.items()):
            axes[i].text(bar.get_x() + bar.get_width() / 2, cnt + 0.5,
                         f"{cnt}\n({cnt/len(sub):.0%})", ha="center",
                         fontsize=9, fontweight="bold")

    fig2.legend(handles=[mpatches.Patch(color=c, label=s) for s, c in SELECTOR_COLORS.items()],
                loc="upper right", fontsize=9, title="Selector")
    fig2.suptitle("Distribucion de best_selector por cluster", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(output_dir / "clustering_selectores.png", dpi=150, bbox_inches="tight")
    return fig1, fig2


def plot_caracterizacion_clusters(
    meta_dataset_df: pd.DataFrame,
    labels,
    output_path: str | Path | None = "figures/caracterizacion_clusters.png",
) -> plt.Figure:
    """Plot boxplots for interpretable dataset statistics by cluster."""
    df_clusters = meta_dataset_df.copy()
    nombres_cluster = {
        0: "C0 - f_classif",
        1: "C1 - mutual_info",
        2: "C2 - ambiguo",
    }
    df_clusters["cluster"] = labels
    df_clusters["cluster_nombre"] = df_clusters["cluster"].map(nombres_cluster)

    features_plot = [
        ("nr_inst", "N instancias"),
        ("nr_attr", "N features"),
        ("class_ent", "Entropia de clases (bits)"),
        ("majority_class_error", "Error clase mayoritaria"),
        ("class_balance_ratio", "Ratio de balance"),
        ("nr_class", "N clases"),
    ]
    features_plot = [(f, n) for f, n in features_plot if f in df_clusters.columns]

    colores = {
        "C0 - f_classif": "#C44E52",
        "C1 - mutual_info": "#55A868",
        "C2 - ambiguo": "#4C72B0",
    }
    orden = ["C0 - f_classif", "C1 - mutual_info", "C2 - ambiguo"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for i, (feat, nombre) in enumerate(features_plot):
        ax = axes[i]
        sns.boxplot(
            data=df_clusters,
            x="cluster_nombre",
            y=feat,
            order=orden,
            palette=colores,
            ax=ax,
        )
        ax.set_title(nombre, fontweight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(features_plot), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Caracterizacion de datasets por cluster", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig

