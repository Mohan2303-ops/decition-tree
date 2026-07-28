"""
Step 42: Precision-Recall Curve
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #42
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Precision-Recall Curve

PrecisionRecallDisplay.from_predictions(
    y2_test,
    test_prob2,
    name="Heart Disease CART"
)

plt.axhline(
    y2_test.mean(),
    linestyle="--"
)

plt.title("Precision-Recall Curve")

plt.show()
