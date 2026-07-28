"""
Step 54: Rank the Models
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #54
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Rank the Models

ranking = combined_results.copy()

ranking["Average Score"] = ranking[
    [
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Precision",
        "F1 Score",
        "ROC AUC",
        "PR AUC"
    ]
].mean(axis=1)

ranking = ranking.sort_values(
    by="Average Score",
    ascending=False
)

display(ranking)
