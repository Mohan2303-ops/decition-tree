"""
Step 53: Radar Chart (Overall Performance)
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #53
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Radar Chart (Overall Performance)

metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Sensitivity",
    "Precision",
    "F1 Score",
    "ROC AUC",
    "PR AUC"
]

comparison_table = combined_results.set_index(
    ["Dataset","Model"]
)[metrics]

display(comparison_table)
