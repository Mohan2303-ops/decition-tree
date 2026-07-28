"""
Step 16: ROC Curve
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #16
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# ROC Curve

RocCurveDisplay.from_predictions(
    y_test,
    test_prob,
    name="Final CART"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.title("ROC Curve")

plt.show()
