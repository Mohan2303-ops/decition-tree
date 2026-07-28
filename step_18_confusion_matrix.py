"""
Step 18: Confusion Matrix
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #18
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Confusion Matrix

test_prediction = (
    test_prob >= selected_threshold
).astype(int)

ConfusionMatrixDisplay.from_predictions(

    y_test,

    test_prediction,

    display_labels=[
        "Benign",
        "Malignant"
    ],

    cmap="Blues",

    values_format="d"

)

plt.title("Confusion Matrix")

plt.show()
