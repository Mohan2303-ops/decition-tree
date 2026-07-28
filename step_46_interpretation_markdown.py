"""
Step 46: Interpretation (Markdown)
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #46
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Interpretation (Markdown)

# Decision Path 1
# • The model first evaluates the most informative feature (for example, chest pain type or oldpeak).
# • If the feature value falls below the learned threshold, the instance moves toward the "No Disease" branch.
# • Additional conditions are evaluated until a terminal leaf predicts No Heart Disease.

# Decision Path 2
# • If the initial split exceeds the threshold, the model follows the branch associated with higher heart disease risk.
# • Subsequent splits may use features such as thalach, ca, or thal.
# • The instance reaches a terminal leaf predicting Heart Disease.

# Note:
# The split thresholds are learned from the training data and are intended for educational purposes only. They should not be interpreted as clinical decision rules.
