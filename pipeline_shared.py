"""
Common imports shared by every step file.
Each step does `from pipeline_shared import *` so it has access to
numpy / pandas / sklearn / matplotlib without repeating the imports,
and so that `display()` works outside of Jupyter.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_validate, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, RocCurveDisplay,
    PrecisionRecallDisplay,
)

RANDOM_STATE = 42

def display(x):
    """Fallback for notebook-only display() when run as a plain script."""
    print(x)
