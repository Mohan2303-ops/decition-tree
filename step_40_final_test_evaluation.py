"""
Step 40: Final Test Evaluation
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #40
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Final Test Evaluation

# Train Final Model

final_model2 = selected_model2.fit(
    X2_train,
    y2_train
)

test_prob2 = final_model2.predict_proba(
    X2_test
)[:,1]

# Evaluation Function

final_results2 = evaluate_model(
    y2_test,
    test_prob2,
    selected_threshold2
)

display(final_results2)
