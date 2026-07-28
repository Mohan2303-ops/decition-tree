"""
Step 6: Dummy Baseline
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #6
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Dummy Baseline

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

dummy = DummyClassifier(strategy="prior")

dummy_cv = cross_validate(
    dummy,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

dummy_results = pd.DataFrame(dummy_cv).filter(regex="test_")

summary = pd.DataFrame({
    "Mean": dummy_results.mean(),
    "Std": dummy_results.std()
})

display(summary)
