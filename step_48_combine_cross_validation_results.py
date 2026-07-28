"""
Step 48: Combine Cross-Validation Results
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #48
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Combine Cross-Validation Results

partA_comparison = comparison.copy()
partA_comparison["Dataset"] = "Breast Cancer"

partB_comparison = comparison2.copy()
partB_comparison["Dataset"] = "Heart Disease"

combined_results = pd.concat(
    [partA_comparison, partB_comparison],
    ignore_index=True
)

combined_results = combined_results[
    [
        "Dataset",
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Precision",
        "F1 Score",
        "ROC AUC",
        "PR AUC"
    ]
]

display(combined_results)
