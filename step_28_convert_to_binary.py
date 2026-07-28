"""
Step 28: Convert to binary
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #28
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Convert to binary
y2 = y2.astype(int)

# Rename target
y2.name = "HeartDisease"

eda2 = pd.concat([X2, y2], axis=1)

display(eda2.head())
