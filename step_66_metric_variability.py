"""
Step 66: Metric Variability
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #66
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Metric Variability

metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

variability = pd.DataFrame({
    "Mean": stability_df[metrics].mean(),
    "Standard Deviation": stability_df[metrics].std()
})

display(variability)
