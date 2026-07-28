"""
Step 15: Final Test Evaluation
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #15
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Final Test Evaluation

# Fit Final Model

final_model = selected_model.fit(
    X_train,
    y_train
)

test_prob = final_model.predict_proba(
    X_test
)[:,1]

# Evaluation Function

def evaluate_model(
    y_true,
    probability,
    threshold
):

    prediction = (
        probability >= threshold
    ).astype(int)

    tn,fp,fn,tp = confusion_matrix(
        y_true,
        prediction
    ).ravel()

    specificity = tn/(tn+fp)

    results = {

        "TN":tn,

        "FP":fp,

        "FN":fn,

        "TP":tp,

        "Accuracy":
        accuracy_score(
            y_true,
            prediction
        ),

        "Balanced Accuracy":
        balanced_accuracy_score(
            y_true,
            prediction
        ),

        "Sensitivity":
        recall_score(
            y_true,
            prediction
        ),

        "Specificity":
        specificity,

        "Precision":
        precision_score(
            y_true,
            prediction
        ),

        "F1":
        f1_score(
            y_true,
            prediction
        ),

        "ROC AUC":
        roc_auc_score(
            y_true,
            probability
        ),

        "PR AUC":
        average_precision_score(
            y_true,
            probability
        )

    }

    return pd.Series(results)

# Final Metrics

final_results = evaluate_model(
    y_test,
    test_prob,
    selected_threshold
)

display(final_results)
