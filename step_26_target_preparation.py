"""
Step 26: Target Preparation
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #26
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Target Preparation

# Separate features and target
X2 = heart.data.copy()
y2 = X2.pop("target")

# Convert target to integer if necessary
y2 = y2.astype(int)

print("Feature Shape:", X2.shape)
print("Target Shape:", y2.shape)

print("\nTarget Distribution")
print(y2.value_counts())

print(X2.isnull().sum())
