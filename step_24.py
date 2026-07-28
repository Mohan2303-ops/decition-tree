"""
Step 24: Step 24
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #24
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

print("Dataset Shape")

print(X2.shape)

print("\nTarget Shape")

print(y2.shape)

print("\nData Types")

display(X2.dtypes)

print("\nMissing Values")

display(X2.isnull().sum())

print("\nDuplicate Rows")

print(X2.duplicated().sum())

print("\nStatistical Summary")

display(X2.describe(include="all"))
