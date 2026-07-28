"""
Step 52: Balanced Accuracy Comparison
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #52
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Balanced Accuracy Comparison

plt.figure(figsize=(10,6))

sns.barplot(
    data=combined_results,
    x="Model",
    y="Balanced Accuracy",
    hue="Dataset",
    palette="viridis"
)

plt.title("Balanced Accuracy Comparison")

plt.xticks(rotation=15)

plt.show()
