"""
Step 43: Confusion Matrix
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #43
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Confusion Matrix

test_prediction2 = (
    test_prob2 >= selected_threshold2
).astype(int)

ConfusionMatrixDisplay.from_predictions(

    y2_test,

    test_prediction2,

    display_labels=[
        "No Disease",
        "Disease"
    ],

    cmap="Blues",

    values_format="d"

)

plt.title("Confusion Matrix")

plt.show()
