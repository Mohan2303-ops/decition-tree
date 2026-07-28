"""
Step 37: Random Forest
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #37
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Random Forest

rf2 = Pipeline([

    ("preprocessor",preprocessor),

    ("model",RandomForestClassifier(

        n_estimators=500,

        max_features="sqrt",

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    ))
])

# Cross Validation

rf_cv2 = cross_validate(

    rf2,

    X2_train,

    y2_train,

    cv=cv2,

    scoring=scoring,

    return_train_score=True
)

rf_results2 = pd.DataFrame(
    rf_cv2
)

summary_rf2 = pd.DataFrame({

    "Train Mean":rf_results2.filter(regex="train").mean(),

    "Test Mean":rf_results2.filter(regex="test").mean(),

    "Train Std":rf_results2.filter(regex="train").std(),

    "Test Std":rf_results2.filter(regex="test").std()

})

display(summary_rf2)
