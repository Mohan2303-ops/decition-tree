"""
Step 17: Precision Recall Curve
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #17
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Precision Recall Curve

PrecisionRecallDisplay.from_predictions(
    y_test,
    test_prob,
    name="Final CART"
)

plt.axhline(
    y_test.mean(),
    linestyle="--"
)

plt.title("Precision Recall Curve")

plt.show()
