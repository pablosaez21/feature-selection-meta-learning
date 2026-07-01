# Configuración inicial
SEED = 42

# 1. Librerías estándar de Python
import gc  # Garbage collector para liberar memoria activamente
import os
import time

# 2. Manejo de datos y utilidades
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# 3. Visualización
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# 4. Librerías específicas de datos y meta-características
import openml
from pymfe.mfe import MFE

# 5. Scikit-learn (Machine Learning) agrupado por submódulos
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

