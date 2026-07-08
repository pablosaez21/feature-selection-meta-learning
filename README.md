# Meta-Learning for Feature Selection

## Description

This repository contains the code and notebook for a Bachelor's Thesis project on meta-learning for automated recommendation of univariate feature selection filters in classification problems.

The project builds a meta-dataset from OpenML classification datasets, evaluates several feature selection filters on each dataset, extracts meta-features with PyMFE and custom descriptors, trains meta-learners, and analyzes the resulting meta-dataset through feature-importance and clustering experiments.

## Highlights

- Bachelor's Thesis project on meta-learning for automated feature selector recommendation.
- Built a meta-dataset from 126 OpenML classification datasets.
- Extracted 80 usable meta-features using PyMFE and custom descriptors.
- Evaluated Chi-Square, Mutual Information and ANOVA F-test.
- Best result: Gradient Boosting with Top 30 meta-features reached 0.5873 LOO accuracy vs 0.4127 majority baseline.
- Main value: the model improves over a simple baseline in a difficult low-data meta-learning setting.

## Problem

There is no universally best feature selection filter. The behavior of a filter depends on the characteristics of the dataset: number and type of variables, class distribution, feature-target relationships, noise, redundancy, and classification complexity. A filter that performs well on one dataset may not be the best option for another.

Meta-learning is a suitable approach because it represents each dataset through meta-features and learns relationships between dataset characteristics and selector performance. Instead of manually choosing a filter, the method uses previous evaluations across datasets to recommend a filter for a new classification task.

## Methodology

The experimental pipeline is:

```text
OpenML -> preprocessing -> filter evaluation -> composite score -> meta-feature extraction -> meta-dataset construction -> meta-learner training -> clustering analysis
```

OpenML tasks are collected from the suites used in the notebook. Each dataset is preprocessed, the feature selectors are evaluated using stratified cross-validation, and a composite score is computed from accuracy, standard deviation, and execution time. PyMFE is then used to extract meta-features, which are combined with the selector scores to form the final meta-dataset. The notebook evaluates meta-learners with Leave-One-Out validation and later performs feature-importance and clustering analyses.

The composite score used in the notebook is:

```text
score = 0.7 * accuracy - 0.2 * std - 0.1 * time
```

## Evaluated Feature Selectors

- Chi-Square
- Mutual Information
- ANOVA F-test

## Meta-features

The meta-features are extracted with PyMFE using the following groups:

| Group | Description |
|---|---|
| General | Basic dataset descriptors such as number of instances, attributes, classes, numerical variables, categorical variables, and binary variables |
| Statistical | Distributional and correlation-based descriptors such as skewness, kurtosis, correlations, class balance, and variation coefficients |
| Information-theoretic | Entropy, mutual information, joint entropy, class concentration, and related measures |
| Complexity | Measures that describe classification difficulty and class separability |
| Landmarking | Performance of simple learning algorithms used as dataset descriptors |
| Model-based | Descriptors extracted from decision-tree structure and variable importance |

The final modeling matrix used in the notebook contains 80 usable meta-features.

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

The best reported result is `0.5873` Leave-One-Out accuracy using Gradient Boosting with the Top 30 ranked meta-features. This is not a high absolute accuracy, and it should not be interpreted as a definitive solution to feature selector recommendation. However, it is meaningful in this exploratory setting because it improves over the majority-class baseline of `0.4127` by about 17.5 percentage points.

Target distribution:

| Selector | Count | Proportion |
|---|---:|---:|
| mutual_info | 52 | 41.27% |
| chi2 | 42 | 33.33% |
| f_classif | 32 | 25.40% |

Leave-One-Out results:

| Model | LOO Accuracy |
|---|---:|
| Majority baseline | 0.4127 |
| Direct RandomForest classifier | 0.5079 |
| Hybrid classifier + regression model | 0.5159 |
| Soft-voting ensemble | 0.5476 |
| Gradient Boosting Top-30 meta-features | 0.5873 |

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

## What this project demonstrates

- Machine learning experimentation
- Meta-learning
- Feature selection
- OpenML data handling
- Cross-validation and Leave-One-Out evaluation
- Python modularization
- Result analysis and academic reporting

## Limitations

- The meta-dataset is small for a meta-learning problem.
- The results are exploratory.
- More datasets and more feature selectors would be needed for stronger conclusions.
- The project is scalable and parametrizable, but it is research-oriented in its current form.
- With more computational resources, the pipeline could be extended and prepared for software-oriented use; in its current state, it is not production-ready.

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
- `src/`: contains the notebook code separated into Python files by experimental block.
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

The notebook contains the full experimental workflow: OpenML loading, preprocessing, selector evaluation, meta-feature extraction, meta-dataset construction, meta-learner evaluation, feature-importance analysis, and clustering analysis.

## Technologies Used

Python, scikit-learn, OpenML, PyMFE, pandas, NumPy, Matplotlib/Seaborn and Jupyter Notebook.
