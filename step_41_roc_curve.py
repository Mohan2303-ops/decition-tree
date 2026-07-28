"""
Step 41: ROC Curve
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #41
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# ROC Curve

RocCurveDisplay.from_predictions(
    y2_test,
    test_prob2,
    name="Heart Disease CART"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.title("ROC Curve")

plt.show()
