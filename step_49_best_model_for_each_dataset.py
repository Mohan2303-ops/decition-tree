"""
Step 49: Best Model for Each Dataset
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #49
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Best Model for Each Dataset

best_models = combined_results.loc[
    combined_results.groupby("Dataset")["ROC AUC"].idxmax()
]

display(best_models)
