"""
Step 3: Data Audit
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #3
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Data Audit

print("Shape of Dataset")

display(X.shape)

print("\nData Types")

display(X.dtypes)

print("\nMissing Values")

display(X.isnull().sum())

print("\nDuplicate Rows")

print(X.duplicated().sum())

print("\nUnique Values")

display(X.nunique())

print("\nStatistical Summary")

display(X.describe())


# Leakage Check

print("Checking for Target Leakage")

assert "target" not in X.columns
assert "Malignant" not in X.columns

print("No Target Leakage Found")
