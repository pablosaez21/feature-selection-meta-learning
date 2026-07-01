"""Visualization helpers for the notebook analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


GROUP_COLORS = {
    "General": "#4C72B0",
    "Statistical": "#DD8452",
    "Information-theoretic": "#55A868",
    "Complexity": "#C44E52",
    "Landmarking": "#8172B3",
    "Model-based": "#937860",
}

SELECTOR_COLORS = {
    "chi2": "#DD8452",
    "mutual_info": "#55A868",
    "f_classif": "#8172B3",
}


def plot_feature_ablation(
    ablation_df: pd.DataFrame,
    output_path: str | Path | None = "figures/feature_ablation.png",
):
    """Plot RF and GB Leave-One-Out accuracy by top-k feature cutoff."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        ablation_df["n_features"],
        ablation_df["random_forest_accuracy"],
        "o-",
        color="#4C72B0",
        label="RandomForest",
    )
    ax.plot(
        ablation_df["n_features"],
        ablation_df["gradient_boosting_accuracy"],
        "s-",
        color="#C44E52",
        label="GradientBoosting",
    )
    ax.axhline(y=0.4127, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(x=30, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("LOO Accuracy")
    ax.set_title("Effect of the number of meta-features")
    ax.set_xticks(ablation_df["n_features"])
    ax.set_ylim(0.38, 0.62)
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_top_importances(importances: pd.Series, top_n: int = 25):
    """Plot the top RandomForest meta-feature importances."""
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.head(top_n).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top meta-features by RandomForest importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_cluster_projection(
    labels,
    X_2d,
    output_path: str | Path | None = "figures/clustering_structure.png",
):
    """Plot the PCA projection of KMeans clusters."""
    cluster_colors = {0: "#C44E52", 1: "#55A868", 2: "#4C72B0"}
    cluster_names = {
        0: "Cluster 0 - f_classif",
        1: "Cluster 1 - mutual_info",
        2: "Cluster 2 - ambiguous",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id, color in cluster_colors.items():
        mask = labels == cluster_id
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=color,
            label=cluster_names[cluster_id],
            alpha=0.75,
            s=60,
            edgecolors="white",
            linewidth=0.5,
        )
    ax.set_title("Cluster structure in the meta-dataset")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_selector_distribution_by_cluster(
    meta_dataset_df: pd.DataFrame,
    labels,
    output_path: str | Path | None = "figures/clustering_selectors.png",
):
    """Plot best-selector distribution inside each cluster."""
    df = meta_dataset_df.copy()
    df["cluster"] = labels
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for i, cluster_id in enumerate([0, 1, 2]):
        subset = df[df["cluster"] == cluster_id]
        counts = subset["best_selector"].value_counts()
        bars = axes[i].bar(
            counts.index,
            counts.values,
            color=[SELECTOR_COLORS[s] for s in counts.index],
            edgecolor="black",
            linewidth=1,
        )
        axes[i].set_title(f"Cluster {cluster_id} ({len(subset)} datasets)")
        axes[i].grid(axis="y", alpha=0.3)
        for bar, (_, count) in zip(bars, counts.items()):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                count + 0.5,
                f"{count}\n({count / len(subset):.0%})",
                ha="center",
                fontsize=9,
            )

    fig.legend(
        handles=[mpatches.Patch(color=color, label=name) for name, color in SELECTOR_COLORS.items()],
        loc="upper right",
        fontsize=9,
        title="Selector",
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cluster_characterization(
    meta_dataset_df: pd.DataFrame,
    labels,
    output_path: str | Path | None = "figures/cluster_characterization.png",
):
    """Plot boxplots for interpretable dataset statistics by cluster."""
    df = meta_dataset_df.copy()
    cluster_names = {
        0: "C0 - f_classif",
        1: "C1 - mutual_info",
        2: "C2 - ambiguous",
    }
    df["cluster_name"] = pd.Series(labels).map(cluster_names)

    features = [
        ("nr_inst", "Number of instances"),
        ("nr_attr", "Number of features"),
        ("class_ent", "Class entropy"),
        ("majority_class_error", "Majority-class error"),
        ("class_balance_ratio", "Class balance ratio"),
        ("nr_class", "Number of classes"),
    ]
    features = [(feature, title) for feature, title in features if feature in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    order = ["C0 - f_classif", "C1 - mutual_info", "C2 - ambiguous"]

    for i, (feature, title) in enumerate(features):
        sns.boxplot(data=df, x="cluster_name", y=feature, order=order, ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", rotation=20)
        axes[i].grid(axis="y", alpha=0.3)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig

