"""
Step 27: Convert integer-coded categories to categorical dtype
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #27
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

# Convert integer-coded categories to categorical dtype
X2[categorical_columns] = X2[categorical_columns].astype("category")

print("Updated Data Types")

display(X2.dtypes)
