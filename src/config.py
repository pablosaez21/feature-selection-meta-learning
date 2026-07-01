"""Shared configuration for the experiments."""

SEED = 42

SELECTOR_NAMES = ["chi2", "mutual_info", "f_classif"]

W_ACC = 0.7
W_STD = 0.2
W_TIME = 0.1

MAX_INSTANCES = 30_000
MAX_FEATURES = 1_000
MAX_CLASSES = 50

OPENML_SUITES = ["OpenML-CC18", "14", "270", "334", "379"]

MFE_GROUPS = [
    "general",
    "statistical",
    "info-theory",
    "complexity",
    "landmarking",
    "model-based",
]

MFE_SUMMARIES = ["mean", "sd", "min", "max"]

