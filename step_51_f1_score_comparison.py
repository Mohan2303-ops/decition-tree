"""
Step 51: F1 Score Comparison
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #51
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# F1 Score Comparison

plt.figure(figsize=(10,6))

sns.barplot(
    data=combined_results,
    x="Model",
    y="F1 Score",
    hue="Dataset",
    palette="Set1"
)

plt.title("F1 Score Comparison")

plt.xticks(rotation=15)

plt.show()
