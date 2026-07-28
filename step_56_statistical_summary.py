"""
Step 56: Statistical Summary
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #56
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Statistical Summary

stats = combined_results.groupby(
    "Dataset"
)[
    [
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Precision",
        "F1 Score",
        "ROC AUC",
        "PR AUC"
    ]
].mean()

display(stats)
