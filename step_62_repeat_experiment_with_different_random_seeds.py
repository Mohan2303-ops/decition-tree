"""
Step 62: Repeat Experiment with Different Random Seeds
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #62
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Repeat Experiment with Different Random Seeds

seeds = [7, 21, 42, 84, 100]

stability_results = []

for seed in seeds:

    X_train_seed, X_test_seed, y_train_seed, y_test_seed = train_test_split(
        X2,
        y2,
        test_size=0.20,
        stratify=y2,
        random_state=seed
    )

    best_model.fit(
        X_train_seed,
        y_train_seed
    )

    prediction = best_model.predict(X_test_seed)

    probability = best_model.predict_proba(
        X_test_seed
    )[:,1]

    stability_results.append({

        "Random Seed": seed,

        "Accuracy": accuracy_score(
            y_test_seed,
            prediction
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test_seed,
            prediction
        ),

        "Precision": precision_score(
            y_test_seed,
            prediction
        ),

        "Recall": recall_score(
            y_test_seed,
            prediction
        ),

        "F1 Score": f1_score(
            y_test_seed,
            prediction
        ),

        "ROC AUC": roc_auc_score(
            y_test_seed,
            probability
        )

    })

stability_df = pd.DataFrame(stability_results)

display(stability_df)
