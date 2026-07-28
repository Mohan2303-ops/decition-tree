"""
Step 61: Select Best Model
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #61
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Select Best Model

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])
