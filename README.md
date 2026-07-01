# Meta-Learning for Feature Selection

This project explores whether dataset characteristics can be used to recommend the most suitable univariate feature selection filter for a classification problem.

<<<<<<< Updated upstream
The core idea is to treat feature selector recommendation as a **meta-learning task**: each dataset becomes one meta-example, described by meta-features extracted with PyMFE, and labelled with the filter that achieved the best multi-criteria score.

The project was developed as my Bachelor's Thesis in Computer Science and focuses on three univariate feature selection filters:
=======
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

## Evaluated Feature Selectors
>>>>>>> Stashed changes

- Chi-Square
- Mutual Information
- ANOVA F-test

---

## Description

Feature selection is a common preprocessing step in machine learning. Its goal is to identify the most relevant variables in a dataset, reducing dimensionality and potentially improving model performance.

However, there is no universally best feature selection method. A filter that performs well on one dataset may perform poorly on another. This project investigates whether the properties of a dataset can help predict which univariate filter is more suitable.

The final goal is not to build a production-ready recommendation system, but to study whether there is enough signal in dataset meta-features to support automatic feature selector recommendation.

---

## Main Contribution

The main contribution of this project is an exploratory meta-learning pipeline that shows measurable signal in the relationship between dataset properties and feature selector performance.

The best model, **Gradient Boosting trained on the Top 30 ranked meta-features**, achieved a Leave-One-Out accuracy of **0.5873**, compared with a majority-class baseline of **0.4127**.

This represents an improvement of approximately **+17.5 percentage points** over the baseline.

This result suggests that the extracted meta-features contain useful information for recommending feature selection filters. However, the problem remains difficult due to the small number of meta-examples, the limited computational budget and the moderate separability between selectors.

---

## Problem

The project is based on the idea that no single feature selection filter is optimal for every dataset.

Given a new classification dataset, the question is:

> Can we predict which univariate feature selection filter will work best by looking only at the characteristics of the dataset?

To answer this, the project builds a meta-dataset where:

- each row represents a dataset from OpenML;
- the input variables are meta-features describing that dataset;
- the target is the feature selector that achieved the best composite score.

---

## Methodology

The full experimental pipeline is:

```text
OpenML datasets
        ↓
Preprocessing and missing value handling
        ↓
Evaluation of Chi-Square, Mutual Information and ANOVA F-test
        ↓
Composite score calculation
        ↓
Meta-feature extraction with PyMFE
        ↓
Meta-dataset construction
        ↓
Meta-learner training and evaluation
        ↓
Feature importance, ablation and clustering analysis
```

For each dataset, the three filters are evaluated using stratified cross-validation. Their performance is summarized with a composite score based on three criteria:

- mean accuracy;
- standard deviation across folds;
- execution time.

The best-scoring selector becomes the target label for that dataset.

---

## Evaluated Feature Selectors

The project evaluates three univariate filter methods.

### Chi-Square

Chi-Square measures the statistical dependence between each feature and the target class. It is especially suitable for non-negative and discrete-like feature values.

### Mutual Information

Mutual Information measures how much information a feature provides about the target class. Unlike ANOVA F-test, it can capture non-linear dependencies.

### ANOVA F-test

ANOVA F-test measures whether the mean value of a continuous feature differs significantly across classes. It is useful when the relationship between feature and class is approximately linear.

---

## Composite Score

Using accuracy alone can lead to ties or unstable decisions. For that reason, each selector is evaluated with a composite score:

```text
score = 0.7 · accuracy - 0.2 · std - 0.1 · time
```

Where:

- `accuracy` is the mean cross-validation accuracy;
- `std` is the standard deviation across folds;
- `time` is the execution time.

The weights used were:

| Criterion | Weight |
|---|---:|
| Accuracy | 0.7 |
| Stability | 0.2 |
| Runtime | 0.1 |

The objective is to prioritize predictive performance while also considering stability and computational cost.

---

## Meta-features

Meta-features were extracted using PyMFE. The following groups were used:

| Group | Description |
|---|---|
| General | Basic dataset properties such as number of instances, attributes and classes |
| Statistical | Distributional properties such as skewness, kurtosis and correlations |
| Information-theoretic | Entropy and mutual information based descriptors |
| Complexity | Measures related to class separability and problem difficulty |
| Landmarking | Performance of simple models used as dataset descriptors |
| Model-based | Structural properties extracted from decision-tree models |

In addition, custom meta-features related to class imbalance were included, such as:

- class balance ratio;
- majority class error.

After preprocessing and cleaning, the final meta-dataset contained **126 datasets** and **80 usable meta-features**.

---

The final modeling matrix used in the notebook contains 80 usable meta-features.

## Models

The main meta-learning models evaluated were:

- Direct Random Forest classifier.
- Hybrid classification-regression model.
- Soft-voting ensemble.
- Gradient Boosting evaluated over Top-k ranked meta-features.

Additional analyses included:

- Random Forest meta-feature importance;
- feature ablation;
- learning curve analysis;
- PCA projection;
- KMeans clustering;
- Ward hierarchical clustering.

---

## Results

The final target distribution was moderately imbalanced:

<<<<<<< Updated upstream
| Selector | Count | Proportion |
|---|---:|---:|
| mutual_info | 52 | 41.27% |
| chi2 | 42 | 33.33% |
| f_classif | 32 | 25.40% |
=======
The best reported result is `0.5873` Leave-One-Out accuracy using Gradient Boosting with the Top 30 ranked meta-features. This is not a high absolute accuracy, and it should not be interpreted as a definitive solution to feature selector recommendation. However, it is meaningful in this exploratory setting because it improves over the majority-class baseline of `0.4127` by about 17.5 percentage points.

Target distribution:
>>>>>>> Stashed changes

The majority-class baseline is therefore **0.4127**, obtained by always predicting `mutual_info`.

### Leave-One-Out Evaluation

| Model | LOO Accuracy | Improvement vs Baseline |
|---|---:|---:|
| Majority baseline | 0.4127 | — |
| Direct Random Forest classifier | 0.5079 | +9.5 pp |
| Hybrid classifier + regression model | 0.5159 | +10.3 pp |
| Soft-voting ensemble | 0.5476 | +13.5 pp |
| Gradient Boosting Top-30 meta-features | **0.5873** | **+17.5 pp** |

The strongest result was obtained after reducing the meta-feature space to the 30 most relevant descriptors. This is important because the full meta-dataset has a high dimensionality relative to the number of datasets, making overfitting a central risk.

---

## Meta-feature Selection Effect

The feature ablation experiment showed that using all available meta-features was not optimal.

Gradient Boosting achieved its best result with the Top 30 ranked meta-features:

| Top-k meta-features | Random Forest | Gradient Boosting |
|---:|---:|---:|
| 30 | 0.5635 | **0.5873** |
| 40 | 0.5238 | 0.5397 |
| 50 | 0.5159 | 0.5317 |
| 60 | 0.5238 | 0.5000 |
| 70 | 0.5317 | 0.5397 |
| 80 | 0.4921 | 0.5317 |

This suggests that the meta-learning problem benefits from reducing noisy or redundant meta-features, especially given the limited number of available meta-examples.

---

## Clustering Analysis

A clustering analysis was performed to inspect whether the meta-dataset contained groups of datasets with different selector behavior.

Gradient Boosting Top-30 accuracy by cluster:

| Cluster | Description | Datasets | Correct | Accuracy |
|---:|---|---:|---:|---:|
| 0 | f_classif-oriented, medium confidence | 31 | 13 | 0.4194 |
| 1 | mutual_info-oriented, high confidence | 26 | 20 | **0.7692** |
| 2 | ambiguous, low confidence | 69 | 41 | 0.5942 |

<<<<<<< Updated upstream
The strongest performance appears in Cluster 1, where `mutual_info` dominates more clearly. This supports the idea that meta-learning is more effective in regions of the meta-dataset where selector behavior is structurally more consistent.

This analysis is valuable because it shows that the recommendation problem is not equally difficult across all datasets. Some groups contain clearer patterns, while others are more ambiguous.

---

## Key Findings

The main findings of the project are:

- The meta-learners consistently outperform the majority-class baseline.
- The best result is obtained with Gradient Boosting using the Top 30 meta-features.
- Reducing the meta-feature space improves performance, suggesting that not all meta-features are equally useful.
- The clustering analysis reveals regions of the meta-dataset where selector recommendation is easier.
- The results provide evidence that dataset characteristics contain useful signal for feature selector recommendation, although the task remains challenging.

---

## Limitations

This project should be interpreted as an exploratory study rather than a definitive feature selector recommender.

Main limitations:

- The meta-dataset contains only 126 datasets, which is small for an 80-dimensional meta-feature space.
- Some PyMFE meta-features are computationally expensive, which limited the number and size of OpenML datasets that could be processed.
- The recommendation task has three relatively competitive classes, so moderate accuracy is expected.
- The composite score depends on manually chosen weights: 0.7 for accuracy, 0.2 for stability and 0.1 for runtime.
- The results indicate useful signal, but not enough evidence to claim a fully general recommender.

---
=======
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
- The project is research-oriented and is not intended as production software.
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
### Folders

| Path | Description |
|---|---|
| `notebooks/` | Main experimental notebook |
| `src/` | Modular Python code extracted from the notebook |
| `results/` | Output files generated by the experiments |
| `figures/` | Figures used for analysis and documentation |

---
=======
- `notebooks/`: contains the original experimental notebook.
- `src/`: contains the notebook code separated into Python files by experimental block.
- `results/`: intended for generated meta-datasets, intermediate results, and exported tables.
- `figures/`: intended for generated plots.
>>>>>>> Stashed changes

## How to Run

Clone the repository:

```bash
git clone https://github.com/pablosaez21/feature-selection-meta-learning.git
cd feature-selection-meta-learning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/Notebook_final.ipynb
```

<<<<<<< Updated upstream
Due to the computational cost of extracting meta-features across many OpenML datasets, the full experiment may take several hours. Some parts of the pipeline can be expensive, especially complexity and landmarking meta-features.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- OpenML
- PyMFE
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Context

This project was developed as a Bachelor's Thesis in Computer Science.

The work combines concepts from:

- feature selection;
- meta-learning;
- automated machine learning;
- experimental machine learning;
- dataset characterization;
- clustering analysis.

---

## Author

**Pablo Sáez Morales**

Computer Science graduate focused on machine learning, applied AI and data-driven systems.
=======
The notebook contains the full experimental workflow: OpenML loading, preprocessing, selector evaluation, meta-feature extraction, meta-dataset construction, meta-learner evaluation, feature-importance analysis, and clustering analysis.

## Technologies Used

Python, scikit-learn, OpenML, PyMFE, pandas, NumPy, Matplotlib/Seaborn and Jupyter Notebook.
>>>>>>> Stashed changes
