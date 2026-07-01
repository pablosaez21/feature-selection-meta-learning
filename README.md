# Meta-learning para recomendar filtros univariables

Proyecto de TFG sobre meta-aprendizaje para recomendar filtros univariables de seleccion de caracteristicas (`chi2`, `mutual_info_classif`, `f_classif`) a partir de meta-features de datasets de OpenML.

El notebook original no se ha modificado. La logica experimental se ha extraido a `src/` para que el proyecto sea mas legible como repositorio de GitHub/portfolio.

## Estructura

```text
.
├── README.md
├── requirements.txt
├── notebooks/
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

## Uso basico

```python
from src.meta_dataset import construir_meta_dataset
from src.data_loading import guardar_meta_dataset

meta_dataset_df = construir_meta_dataset()
guardar_meta_dataset(meta_dataset_df, "results/meta_dataset_df2")
```

Para reutilizar un meta-dataset ya generado:

```python
from src.data_loading import cargar_meta_dataset
from src.meta_learners import (
    evaluar_clasificador_loo,
    evaluar_hibrido_loo,
    evaluar_ensemble_loo,
    calcular_importancia_meta_features,
    evaluar_ablacion_features,
)

meta_dataset_df = cargar_meta_dataset("results/meta_dataset_df2.pkl")

acc_clf, resultados_clf = evaluar_clasificador_loo(meta_dataset_df)
acc_hibrido, pred_hibrido = evaluar_hibrido_loo(meta_dataset_df)
acc_ens, acc_individual, resultados_ens = evaluar_ensemble_loo(meta_dataset_df)

importancias, X_sc, y = calcular_importancia_meta_features(meta_dataset_df)
resultados_ablacion = evaluar_ablacion_features(X_sc, y, importancias.index.tolist())
```

## Correspondencia notebook -> modulos

- Configuracion inicial, constantes, selectores, pesos y suites OpenML: `src/config.py`.
- Imputacion de datasets y preparacion de matrices: `src/preprocessing.py`.
- Evaluacion de `chi2`, `mutual_info_classif`, `f_classif` con `StratifiedKFold`: `src/selector_evaluation.py`.
- Extraccion de meta-features con PyMFE y grupos originales: `src/meta_features.py`.
- Obtencion de tareas OpenML, guardado y carga del meta-dataset: `src/data_loading.py`.
- Construccion completa del meta-dataset: `src/meta_dataset.py`.
- Validacion Leave-One-Out de meta-learners, enfoque hibrido, ensemble, importancia y ablacion: `src/meta_learners.py`.
- Clustering k=3, PCA y analisis por cluster: `src/clustering_analysis.py`.
- Graficas de importancia, ablacion y clustering: `src/visualization.py`.

## Reproducibilidad

Se mantiene `random_state=42` donde el notebook lo usaba. Tambien se conservan:

- `StratifiedKFold` para evaluar selectores.
- Pesos del score compuesto: accuracy `0.7`, desviacion tipica `0.2`, tiempo `0.1`.
- Validacion Leave-One-Out de los meta-learners.
- `KMeans(n_clusters=3, random_state=42, n_init=20)` para clustering.

