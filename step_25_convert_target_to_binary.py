"""
Step 25: Convert target to binary
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #25
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Convert target to binary

y2 = y2.astype(int)
y2.name = "HeartDisease"

print(y2.value_counts())

print("\n0 = No Heart Disease")
print("1 = Heart Disease")


print(y2.unique())
print(y2.value_counts())
print(y2.dtype)
