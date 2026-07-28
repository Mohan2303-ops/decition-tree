"""
Step 64: Accuracy Across Random Seeds
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #64
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Accuracy Across Random Seeds

plt.figure(figsize=(8,5))

plt.plot(
    stability_df["Random Seed"],
    stability_df["Accuracy"],
    marker="o"
)

plt.title("Accuracy Across Different Random Seeds")

plt.xlabel("Random Seed")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()
