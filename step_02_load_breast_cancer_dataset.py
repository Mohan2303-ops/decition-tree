"""
Step 2: Load Breast Cancer Dataset
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #2
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Load Breast Cancer Dataset

# Load Dataset - 1

raw = load_breast_cancer(as_frame=True)

X = raw.data.copy()

# Make Malignant = 1

y = (raw.target == 0).astype(int)

y.name = "Malignant"

print("Dataset Loaded Successfully\n")

print("Feature Matrix Shape :", X.shape)

print("Target Shape :", y.shape)

print("\nPositive Class")

print("1 = Malignant")
print("0 = Benign")
