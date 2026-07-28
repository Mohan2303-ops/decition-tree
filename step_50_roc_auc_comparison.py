"""
Step 50: ROC-AUC Comparison
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #50
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# ROC-AUC Comparison

plt.figure(figsize=(10,6))

sns.barplot(
    data=combined_results,
    x="Model",
    y="ROC AUC",
    hue="Dataset",
    palette="Set2"
)

plt.title("ROC-AUC Comparison Across Datasets")

plt.xticks(rotation=15)

plt.show()
