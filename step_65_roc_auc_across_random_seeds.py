"""
Step 65: ROC-AUC Across Random Seeds
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #65
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# ROC-AUC Across Random Seeds

plt.figure(figsize=(8,5))

plt.plot(
    stability_df["Random Seed"],
    stability_df["ROC AUC"],
    marker="o"
)

plt.title("ROC-AUC Across Different Random Seeds")

plt.xlabel("Random Seed")

plt.ylabel("ROC-AUC")

plt.grid(True)

plt.show()
