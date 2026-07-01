

# ─── PESOS DEL SCORE COMPUESTO (parametrizables ) ──────────────

W_ACC    = 0.7  
W_STD    = 0.2   
W_TIEMPO = 0.1   

NOMBRES_SELECTORES = ["chi2", "mutual_info", "f_classif"]
MAX_INSTANCIAS = 30_000   # antes: 20_000
MAX_FEATURES   = 1_000     # antes: 500
MAX_CLASES     = 50        # se mantiene igual

def _normalizar_local(valores: np.ndarray) -> np.ndarray:
    """
    Min-max LOCAL entre los 3 selectores del MISMO dataset.
    Si los 3 valores son iguales → empate técnico → 0.5 a todos.
    """
    rango = valores.max() - valores.min()
    if rango == 0:
        return np.full_like(valores, 0.5, dtype=float)
    return (valores - valores.min()) / rango


def calcular_scores_compuestos(resultados: dict,
                                w_acc: float = W_ACC,
                                w_std: float = W_STD,
                                w_tiempo: float = W_TIEMPO) -> dict:
    """
    Score compuesto por selector dentro de UN dataset.

    score = w_acc * norm(acc) - w_std * norm(std) - w_tiempo * norm(tiempo)

    La normalización es LOCAL entre los 3 selectores del mismo dataset.
    """
    nombres = list(resultados.keys())

    accs    = np.array([resultados[s]["acc_media"] for s in nombres])
    stds    = np.array([resultados[s]["acc_std"]   for s in nombres])
    tiempos = np.array([resultados[s]["tiempo"]    for s in nombres])
    # Normalizamos cada métrica entre 0 y 1 dentro del mismo dataset
    norm_acc    = _normalizar_local(accs)
    norm_std    = _normalizar_local(stds)
    norm_tiempo = _normalizar_local(tiempos)
    
    scores_arr = w_acc * norm_acc - w_std * norm_std - w_tiempo * norm_tiempo

    return dict(zip(nombres, scores_arr))


def preparar_X_e_y(X_df, y):
    """
    Adapta el dataset al formato que necesitan los selectores y el clasificador.
    - Variables categóricas → one-hot encoding (sklearn solo trabaja con números)
    - Variable objetivo → label encoding si no es ya numérica
    """
    X = pd.get_dummies(X_df, drop_first=False)

    y = pd.Series(y).copy()
    if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y


def evaluar_selectores_en_dataset(X_df, y, seed=SEED):
    """
    Evalúa los 3 selectores con CV de 5 folds y devuelve scores compuestos.
    Las métricas brutas (acc, std, tiempo) NO se exponen al meta-dataset:
    se usan solo internamente para calcular el score compuesto.
    """
    X, y = preparar_X_e_y(X_df, y)
     # k = número de features a seleccionar: raíz cuadrada del total,
    # con un mínimo de 2 para que el selector tenga sentido
    k = max(2, int(np.sqrt(X.shape[1])))

    selectores_config = {
        "chi2":        {"score_func": chi2,                "usar_minmax": True},
        "mutual_info": {"score_func": mutual_info_classif, "usar_minmax": False},
        "f_classif":   {"score_func": f_classif,           "usar_minmax": False},
    }

    resultados = {}
  
    # n_splits no puede superar el número de muestras de la clase minoritaria
    min_samples_clase = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_samples_clase)
    if n_splits < 2:
        # Con menos de 2 muestras en alguna clase no tiene sentido evaluar
        raise ValueError(f"Clase minoritaria con solo {min_samples_clase} muestra/s")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for nombre_selector, config in selectores_config.items():
        try:
            if config["usar_minmax"]:
                pipe = Pipeline([
                    ("scaler",   MinMaxScaler()),
                    ("selector", SelectKBest(score_func=config["score_func"], k=k)),
                    ("clf",      RandomForestClassifier(n_estimators=100, random_state=seed)),
                ])
            else:
                pipe = Pipeline([
                    ("selector", SelectKBest(score_func=config["score_func"], k=k)),
                    ("clf",      RandomForestClassifier(n_estimators=100, random_state=seed)),
                ])

            t_inicio = time.time()
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            resultados[nombre_selector] = {
                "acc_media": scores.mean(),
                "acc_std":   scores.std(),
                "tiempo":    time.time() - t_inicio,
            }

        except Exception as e:
            print(f"  ⚠ {nombre_selector} falló en este dataset: {e}")
            resultados[nombre_selector] = {"acc_media": 0.0, "acc_std": 1.0, "tiempo": 1e6}

    scores_compuestos = calcular_scores_compuestos(resultados)
    mejor_selector    = max(scores_compuestos, key=scores_compuestos.get)

    return scores_compuestos, mejor_selector


def procesar_dataset_completo(task_id, seed=SEED):
    """
    Carga un dataset, calcula scores compuestos y extrae meta-features.
    Devuelve UNA fila del meta-dataset.
    """
    try:
        task    = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
        target_name = dataset.default_target_attribute or getattr(task, "target_name", None)
        if target_name is None:
            return None

        X_df, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
        y = pd.Series(y)

        # Filtro de tamaño defensivo
        if X_df.shape[0] > MAX_INSTANCIAS or X_df.shape[1] > MAX_FEATURES:
            print(f"  ↷ Saltado (tamaño real): {dataset.name}")
            return None

        mascara = y.notna().values
        X_df = X_df.loc[mascara].reset_index(drop=True)
        y    = y.loc[mascara].reset_index(drop=True)
        X_df = imputar_dataframe(X_df)

        nombre = f"{dataset.name}_{task_id}"

        scores_compuestos, mejor_selector = evaluar_selectores_en_dataset(X_df, y, seed=seed)
        meta = extraer_meta_features_pymfe(X_df, y)

        fila = {"dataset_name": nombre, "best_selector": mejor_selector}
        fila.update(meta)
        for sel in NOMBRES_SELECTORES:
            fila[f"score_{sel}"] = scores_compuestos[sel]

        return fila

    except Exception as e:
        print(f"  ✗ Error en task {task_id}: {e}")
        return None

    finally:
        try:
            del X_df, y
        except Exception:
            pass
        gc.collect()




# ─── 1) TASKS DE LAS SUITES ─────────────────────────

suite1 = openml.study.get_suite("OpenML-CC18")
suite2 = openml.study.get_suite("14")
suite3 = openml.study.get_suite("270")
suite4 = openml.study.get_suite("334")
suite5 = openml.study.get_suite("379")

task_ids_suites = list(set(
    list(suite1.tasks) +
    list(suite2.tasks) +
    list(suite3.tasks) +
    list(suite4.tasks) +
    list(suite5.tasks)
))
print(f"Tasks totales de las suites (CC18 + 14 + 270 + 334 + 379): {len(task_ids_suites)}")

# ─── 2) UNIÓN FINAL Y BUCLE PRINCIPAL ───────────────────────────────────────
task_ids = task_ids_suites
print(f"Total de tasks a procesar: {len(task_ids)}\n")

filas_meta = []
for i, task_id in enumerate(task_ids, start=1):
    print(f"[{i}/{len(task_ids)}] task_id={task_id}")
    fila = procesar_dataset_completo(task_id)
    if fila is not None:
        filas_meta.append(fila)
        print(f"  ✓ {fila['dataset_name']} — mejor selector: {fila['best_selector']}")

meta_dataset_df = pd.DataFrame(filas_meta)
print(f"\nMeta-dataset final: {meta_dataset_df.shape[0]} datasets × {meta_dataset_df.shape[1]} columnas")


