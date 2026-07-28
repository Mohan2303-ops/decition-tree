"""
Step 12: Random Forest
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #12
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Random Forest

rf = RandomForestClassifier(

    n_estimators=500,

    max_features="sqrt",

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1
)

rf_cv = cross_validate(

    rf,

    X_train,

    y_train,

    cv=cv,

    scoring=scoring,

    return_train_score=True
)

rf_results = pd.DataFrame(rf_cv)

summary_rf = pd.DataFrame({

    "Train Mean":rf_results.filter(regex="train").mean(),

    "Test Mean":rf_results.filter(regex="test").mean(),

    "Train Std":rf_results.filter(regex="train").std(),

    "Test Std":rf_results.filter(regex="test").std()

})

display(summary_rf)
