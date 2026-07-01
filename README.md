# Meta-Learning for Feature Selection

## Description

This project studies a meta-learning approach for recommending univariate feature selection filters for classification datasets. The notebook builds a meta-dataset from OpenML tasks, evaluates several filters on each dataset, extracts meta-features with PyMFE, trains meta-learners, and analyzes the structure of the resulting meta-dataset through clustering.

## Problem

There is no universally best feature selection filter because the usefulness of a filter depends on the statistical properties of the dataset: the number and type of variables, class distribution, feature-target relationships, noise, redundancy, and dataset complexity. A filter that performs well on one dataset may not be the best choice on another.

Meta-learning is suitable for this problem because it represents each dataset through meta-features and learns patterns between dataset characteristics and selector performance. Instead of selecting a filter manually, the method uses previous evaluations across datasets to recommend a filter for a new dataset.

## Methodology

The experimental pipeline is:

OpenML -> preprocessing -> filter evaluation -> composite score -> meta-feature extraction -> meta-dataset construction -> meta-learner training -> clustering analysis.

The OpenML tasks are collected from the suites used in the notebook. Each dataset is preprocessed, the feature selectors are evaluated using stratified cross-validation, and a composite score is computed from accuracy, standard deviation, and execution time. PyMFE is then used to extract meta-features, which are combined with the selector scores to form the final meta-dataset. The notebook evaluates meta-learners with Leave-One-Out validation and later performs feature-importance and clustering analyses.

## Evaluated Feature Selectors

- Chi-Square
- Mutual Information
- ANOVA F-test

## Meta-features

The meta-features are extracted with PyMFE using the following groups:

- General: basic dataset descriptors such as number of instances, attributes, classes, numerical variables, categorical variables, and binary variables.
- Statistical: distributional and correlation-based descriptors such as skewness, kurtosis, correlations, class balance, and variation coefficients.
- Information-theoretic: entropy, mutual information, joint entropy, class concentration, and related measures.
- Complexity: measures that describe classification difficulty and class separability.
- Landmarking: performance of simple learning algorithms used as dataset descriptors.
- Model-based: descriptors extracted from decision-tree structure and variable importance.

## Models

The notebook evaluates several meta-learning approaches:

- Direct RandomForest classifier: predicts the best selector label directly from the meta-features.
- Hybrid model: first classifies `chi2` versus `non-chi2`, then uses Ridge-based multi-output regression to choose between `f_classif` and `mutual_info`.
- Soft-voting ensemble: combines RandomForest, GradientBoosting, LogisticRegression, and KNN using soft voting.
- Feature-importance analysis: trains a RandomForest on the full scaled meta-feature matrix to rank meta-features.
- Feature ablation: evaluates RandomForest and GradientBoosting with different top-k subsets of ranked meta-features.
- Clustering analysis: applies KMeans with `k=3` to the scaled meta-features and studies selector distributions and GradientBoosting accuracy by cluster.

## Results

The notebook output reports a final loaded meta-dataset with 126 datasets and 85 columns. After removing non-feature columns and columns with excessive missing values, the modeling matrix contains 126 datasets and 80 meta-features.

Target distribution:

| Selector | Count |
|---|---:|
| mutual_info | 52 |
| chi2 | 42 |
| f_classif | 32 |

Leave-One-Out results:

| Model | LOO Accuracy |
|---|---:|
| Direct RandomForest classifier | 0.5079 |
| Hybrid classifier + regression model | 0.5159 |
| Soft-voting ensemble | 0.5476 |

Individual models inside the soft-voting ensemble:

| Model | LOO Accuracy |
|---|---:|
| RandomForest | 0.5079 |
| GradientBoosting | 0.5397 |
| LogisticRegression | 0.4921 |
| KNN | 0.4524 |

Feature ablation results:

| Top-k meta-features | RandomForest | GradientBoosting |
|---:|---:|---:|
| 30 | 0.5635 | 0.5873 |
| 40 | 0.5238 | 0.5397 |
| 50 | 0.5159 | 0.5317 |
| 60 | 0.5238 | 0.5000 |
| 70 | 0.5317 | 0.5397 |
| 80 | 0.4921 | 0.5317 |

GradientBoosting Top-30 accuracy by cluster:

| Cluster | Description | Datasets | Correct | Accuracy |
|---:|---|---:|---:|---:|
| 0 | f_classif, medium confidence | 31 | 13 | 0.4194 |
| 1 | mutual_info, high confidence | 26 | 20 | 0.7692 |
| 2 | ambiguous, low confidence | 69 | 41 | 0.5942 |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Notebook_final.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── selector_evaluation.py
│   ├── meta_features.py
│   ├── meta_dataset.py
│   ├── meta_learners.py
│   ├── clustering_analysis.py
│   └── visualization.py
├── results/
└── figures/
```

- `notebooks/`: contains the original experimental notebook.
- `src/`: contains reusable Python modules extracted from the notebook logic.
- `results/`: intended for generated meta-datasets, intermediate results, and exported tables.
- `figures/`: intended for generated plots.

## How to Run

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/Notebook_final.ipynb
```

To build the meta-dataset from Python:

```python
from src.meta_dataset import build_meta_dataset
from src.data_loading import save_meta_dataset

meta_dataset_df = build_meta_dataset()
save_meta_dataset(meta_dataset_df)
```

To reuse a previously generated meta-dataset:

```python
from src.data_loading import load_meta_dataset

meta_dataset_df = load_meta_dataset("results/meta_dataset_df2.pkl")
```

## Technologies Used

Python, scikit-learn, OpenML, PyMFE, pandas, NumPy, Matplotlib/Seaborn and Jupyter Notebook.

