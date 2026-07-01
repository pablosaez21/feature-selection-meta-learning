"""Shared experiment configuration."""

SEED = 42

NOMBRES_SELECTORES = ["chi2", "mutual_info", "f_classif"]

W_ACC = 0.7
W_STD = 0.2
W_TIEMPO = 0.1

MAX_INSTANCIAS = 30_000
MAX_FEATURES = 1_000
MAX_CLASES = 50

OPENML_SUITES = ["OpenML-CC18", "14", "270", "334", "379"]

GRUPOS_MFE = [
    "general",
    "statistical",
    "info-theory",
    "complexity",
    "landmarking",
    "model-based",
]

RESUMENES_MFE = ["mean", "sd", "min", "max"]

