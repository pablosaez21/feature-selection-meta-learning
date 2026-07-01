

# Grupos a usar para meta-features (alineados con paper Parmezan et al. 2021)
GRUPOS_MFE = [
    "general",
    "statistical",
    "info-theory",
    "complexity",
    "landmarking",
    "model-based",
]
# Resúmenes estadísticos que se calculan para cada meta-atributo multivaluado
RESUMENES_MFE = ["mean", "sd", "min", "max"]

# Normaliza el nombre para que no haya caracteres que rompan pandas o sklearn
def limpiar_nombre_meta_feature(nombre):
    return (
        str(nombre)
        .replace(".", "_").replace("-", "_").replace(" ", "_")
        .replace("(", "").replace(")", "").replace("/", "_")
        .replace(":", "_").lower()
    )

# Convierte cualquier valor a float; si no es posible, devuelve NaN
def convertir_a_float(valor):
    try:
        if pd.isna(valor):
            return np.nan
        return float(valor)
    except Exception:
        return np.nan


def extraer_meta_features_pymfe(X_df, y):
    X_df = X_df.copy()
    y = pd.Series(y).reset_index(drop=True)
    
 # Devuelve los índices de las columnas categóricas (texto, categoría, booleano)
 # pymfe necesita saberlo para no tratar esas columnas como numérica
    def _cat_cols(df):
        return [
            i for i, col in enumerate(df.columns)
            if pd.api.types.is_object_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[col])
        ]
     # Función interna que lanza pymfe para un subconjunto de meta-atributos
    def _extraer_bloque(features, summary=None, X_fit_df=None, y_fit=None):
        X_fit_df = X_df if X_fit_df is None else X_fit_df
        y_fit = y if y_fit is None else pd.Series(y_fit).reset_index(drop=True)

        X_fit = X_fit_df.to_numpy()
        cat_cols = _cat_cols(X_fit_df)

        mfe = MFE(
            groups=GRUPOS_MFE,
            features=features,
            summary=summary,
            suppress_warnings=True,
        )
        mfe.fit(X_fit, y_fit.to_numpy(), cat_cols=cat_cols)
        nombres, valores = mfe.extract()

        bloque = {}
        for nombre, valor in zip(nombres, valores):
            nombre_limpio = limpiar_nombre_meta_feature(nombre)
            if isinstance(valor, (list, tuple, np.ndarray)):
                valor = np.asarray(valor).ravel()
                if valor.size == 1:
                    valor = valor.item()
                else:
                    # Si pymfe devuelve un array de más de un elemento, no sabemos
                    # cómo resumirlo aquí → NaN
                    bloque[nombre_limpio] = np.nan
                    continue
            bloque[nombre_limpio] = convertir_a_float(valor)
        return bloque

    meta = {}

    # =========================================================
    # 1) CUSTOM
    # =========================================================
    class_counts = y.value_counts(dropna=False)
    max_class = int(class_counts.max()) if len(class_counts) > 0 else 0
    min_class = int(class_counts.min()) if len(class_counts) > 0 else 0
    
    # Ratio entre la clase minoritaria y la mayoritaria (1.0 = dataset perfectamente balanceado)
    meta["class_balance_ratio"] = (
        convertir_a_float(min_class / max_class) if max_class > 0 else np.nan
    )
    # Error del clasificador naive (predice siempre la clase mayoritaria)
    meta["majority_class_error"] = (
        convertir_a_float(1.0 - (max_class / len(y))) if len(y) > 0 else np.nan
    )

    # =========================================================
    # 2) SIMPLES
    # =========================================================
    try:
        bloque = _extraer_bloque(
            features=["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat", "nr_bin", "attr_to_inst"],
            summary=None,
        )
        for k in ["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat", "nr_bin", "attr_to_inst"]:
            meta[k] = bloque.get(k, np.nan)
    except Exception:
        for k in ["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat", "nr_bin", "attr_to_inst"]:
            meta.setdefault(k, np.nan)

    try:
        bloque = _extraer_bloque(features=["freq_class"], summary=["mean", "sd", "min", "max"])
        for k in ["freq_class_mean", "freq_class_sd", "freq_class_min", "freq_class_max"]:
            meta[k] = bloque.get(k, np.nan)
    except Exception:
        for k in ["freq_class_mean", "freq_class_sd", "freq_class_min", "freq_class_max"]:
            meta.setdefault(k, np.nan)

    # =========================================================
    # 3) STATISTICAL
    # =========================================================
    try:
        bloque = _extraer_bloque(features=["skewness", "kurtosis", "cor"], summary=["mean", "sd", "min", "max"])
        for k in [
            "skewness_mean", "skewness_sd", "skewness_min", "skewness_max",
            "kurtosis_mean", "kurtosis_sd", "kurtosis_min", "kurtosis_max",
            "cor_mean", "cor_sd", "cor_min", "cor_max",
        ]:
            meta[k] = bloque.get(k, np.nan)
    except Exception:
        for k in [
            "skewness_mean", "skewness_sd", "skewness_min", "skewness_max",
            "kurtosis_mean", "kurtosis_sd", "kurtosis_min", "kurtosis_max",
            "cor_mean", "cor_sd", "cor_min", "cor_max",
        ]:
            meta.setdefault(k, np.nan)

    # Coeficiente de variación calculado a mano (más estable que el de pymfe)
    try:
        X_num = X_df.select_dtypes(include=[np.number]).copy()
        if X_num.shape[1] > 0:
            means = X_num.mean(axis=0, skipna=True).replace(0, np.nan)
            stds = X_num.std(axis=0, skipna=True)
            vc = (stds / means).replace([np.inf, -np.inf], np.nan).dropna()
            meta["var_coefficient_mean"] = convertir_a_float(vc.mean()) if len(vc) else np.nan
            meta["var_coefficient_sd"]   = convertir_a_float(vc.std())  if len(vc) else np.nan
            meta["var_coefficient_min"]  = convertir_a_float(vc.min())  if len(vc) else np.nan
            meta["var_coefficient_max"]  = convertir_a_float(vc.max())  if len(vc) else np.nan
        else:
            for k in ["var_coefficient_mean", "var_coefficient_sd", "var_coefficient_min", "var_coefficient_max"]:
                meta[k] = np.nan
    except Exception:
        for k in ["var_coefficient_mean", "var_coefficient_sd", "var_coefficient_min", "var_coefficient_max"]:
            meta.setdefault(k, np.nan)

    try:
        bloque = _extraer_bloque(features=["nr_cor_attr", "nr_outliers"], summary=None)
        meta["nr_cor_attr"] = bloque.get("nr_cor_attr", np.nan)
        meta["nr_outliers"] = bloque.get("nr_outliers", np.nan)
    except Exception:
        for k in ["nr_cor_attr", "nr_outliers"]:
            meta.setdefault(k, np.nan)

    # =========================================================
    # 4) INFO-THEORY
    # =========================================================
    try:
        bloque = _extraer_bloque(features=["class_ent", "eq_num_attr", "ns_ratio"], summary=None)
        meta["class_ent"]   = bloque.get("class_ent", np.nan)
        meta["eq_num_attr"] = bloque.get("eq_num_attr", np.nan)
        meta["ns_ratio"]    = bloque.get("ns_ratio", np.nan)
    except Exception:
        for k in ["class_ent", "eq_num_attr", "ns_ratio"]:
            meta.setdefault(k, np.nan)

    try:
        bloque = _extraer_bloque(features=["attr_ent", "mut_inf", "joint_ent"], summary=["mean", "sd", "min", "max"])
        for k in [
            "attr_ent_mean", "attr_ent_sd", "attr_ent_min", "attr_ent_max",
            "mut_inf_mean", "mut_inf_sd", "mut_inf_min", "mut_inf_max",
            "joint_ent_mean", "joint_ent_sd", "joint_ent_min", "joint_ent_max",
        ]:
            meta[k] = bloque.get(k, np.nan)
    except Exception:
        for k in [
            "attr_ent_mean", "attr_ent_sd", "attr_ent_min", "attr_ent_max",
            "mut_inf_mean", "mut_inf_sd", "mut_inf_min", "mut_inf_max",
            "joint_ent_mean", "joint_ent_sd", "joint_ent_min", "joint_ent_max",
        ]:
            meta.setdefault(k, np.nan)

    try:
        bloque = _extraer_bloque(features=["class_conc"], summary=["mean", "sd", "min", "max"])
        for k in ["class_conc_mean", "class_conc_sd", "class_conc_min", "class_conc_max"]:
            meta[k] = bloque.get(k, np.nan)
    except Exception:
        for k in ["class_conc_mean", "class_conc_sd", "class_conc_min", "class_conc_max"]:
            meta.setdefault(k, np.nan)

     # =========================================================
    # 5) COMPLEXITY
    # =========================================================
    try:
        # Escalares: summary=None funciona porque devuelven un único valor
        bloque = _extraer_bloque(
            features=["density", "c1", "c2", "n1", "t2"],
            summary=None
        )
        meta["density"] = bloque.get("density", np.nan)
        meta["c1"]      = bloque.get("c1",      np.nan)
        meta["c2"]      = bloque.get("c2",      np.nan)
        meta["n1"]      = bloque.get("n1",      np.nan)
        meta["t2"]      = bloque.get("t2",      np.nan)
    except Exception:
        for k in ["density", "c1", "c2", "n1", "t2"]:
            meta.setdefault(k, np.nan)
    
    try:
        # Multi-valor: f1, n2, n3, n4 devuelven arrays → necesitan summary="mean"
        # Con summary=["mean"], pymfe devuelve "f1.mean", "n2.mean"...
        # que limpiar_nombre_meta_feature convierte a "f1_mean", "n2_mean"...
        bloque = _extraer_bloque(
            features=["f1", "n2", "n3", "n4"],
            summary=["mean"]
        )
        meta["f1"]      = bloque.get("f1_mean",  np.nan)
        meta["n2_mean"] = bloque.get("n2_mean",  np.nan)
        meta["n3_mean"] = bloque.get("n3_mean",  np.nan)
        meta["n4"]      = bloque.get("n4_mean",  np.nan)
    except Exception:
        for k in ["f1", "n2_mean", "n3_mean", "n4"]:
            meta.setdefault(k, np.nan)
    
    # =========================================================
    # 6) LANDMARKING
    # =========================================================
    try:
        # Multi-valor: cada landmark devuelve accuracy por fold → summary="mean"
        # Con summary=["mean"], pymfe devuelve "best_node.mean" → "best_node_mean"
        # que coincide exactamente con las claves que ya usamos abajo
        bloque = _extraer_bloque(
            features=[
                "best_node", "random_node", "linear_discr",
                "elite_nn", "naive_bayes", "one_nn", "worst_node",
            ],
            summary=["mean"],
        )
        meta["best_node_mean"]    = bloque.get("best_node_mean",    np.nan)
        meta["random_node_mean"]  = bloque.get("random_node_mean",  np.nan)
        meta["linear_discr_mean"] = bloque.get("linear_discr_mean", np.nan)
        meta["elite_nn_mean"]     = bloque.get("elite_nn_mean",     np.nan)
        meta["naive_bayes_mean"]  = bloque.get("naive_bayes_mean",  np.nan)
        meta["one_nn_mean"]       = bloque.get("one_nn_mean",       np.nan)
        meta["worst_node_mean"]   = bloque.get("worst_node_mean",   np.nan)
    except Exception:
        for k in [
            "best_node_mean", "random_node_mean", "linear_discr_mean",
            "elite_nn_mean", "naive_bayes_mean", "one_nn_mean", "worst_node_mean",
        ]:
            meta.setdefault(k, np.nan)
    # =========================================================
    # 7) MODEL-BASED
    # =========================================================
    try:
        bloque = _extraer_bloque(features=["nodes", "leaves"], summary=None)
        meta["nodes"] = bloque.get("nodes", np.nan)
        meta["leaves"] = bloque.get("leaves", np.nan)
    except Exception:
        for k in ["nodes", "leaves"]:
            meta.setdefault(k, np.nan)

    try:
        bloque = _extraer_bloque(
            features=["tree_depth", "leaves_branch", "nodes_per_level", "var_importance"],
            summary=["mean", "max"],
        )

        meta["tree_depth_mean"] = bloque.get("tree_depth_mean", np.nan)
        meta["tree_depth_max"] = bloque.get("tree_depth_max", np.nan)

        meta["leaves_branch_mean"] = bloque.get("leaves_branch_mean", np.nan)
        meta["leaves_branch_max"] = bloque.get("leaves_branch_max", np.nan)

        meta["nodes_per_level_mean"] = bloque.get("nodes_per_level_mean", np.nan)
        meta["nodes_per_level_max"] = bloque.get("nodes_per_level_max", np.nan)

        meta["var_importance_mean"] = bloque.get("var_importance_mean", np.nan)
        meta["var_importance_max"] = bloque.get("var_importance_max", np.nan)
    except Exception:
        for k in [
            "tree_depth_mean", "tree_depth_max",
            "leaves_branch_mean", "leaves_branch_max",
            "nodes_per_level_mean", "nodes_per_level_max",
            "var_importance_mean", "var_importance_max",
        ]:
            meta.setdefault(k, np.nan)


    
    try:
        bloque = _extraer_bloque(features=["leaves_per_class"], summary=["mean", "sd", "min", "max"])
        for k in [
            "leaves_per_class_mean",
            "leaves_per_class_sd",
            "leaves_per_class_min",
            "leaves_per_class_max",
        ]:
            meta[k] = bloque.get(k, np.nan)
    except Exception:
        for k in [
            "leaves_per_class_mean",
            "leaves_per_class_sd",
            "leaves_per_class_min",
            "leaves_per_class_max",
        ]:
            meta.setdefault(k, np.nan)

    return meta
