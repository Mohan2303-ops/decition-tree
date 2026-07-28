"""
Step 55: Cross-Dataset Summary Table
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #55
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Cross-Dataset Summary Table

summary = combined_results.pivot_table(
    index="Model",
    columns="Dataset",
    values="ROC AUC"
)

display(summary)
