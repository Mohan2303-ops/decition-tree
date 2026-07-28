"""
Step 38: Model Comparison
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #38
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Model Comparison

comparison2 = pd.DataFrame({

    "Model":[

        "Dummy",

        "Basic CART",

        "Tuned & Pruned CART",

        "Random Forest"
    ],

    "Accuracy":[

        dummy_results2["test_accuracy"].mean(),

        basic_results2["test_accuracy"].mean(),

        pruned_results2["test_accuracy"].mean(),

        rf_results2["test_accuracy"].mean()
    ],

    "Balanced Accuracy":[

        dummy_results2["test_balanced_accuracy"].mean(),

        basic_results2["test_balanced_accuracy"].mean(),

        pruned_results2["test_balanced_accuracy"].mean(),

        rf_results2["test_balanced_accuracy"].mean()
    ],

    "Sensitivity":[

        dummy_results2["test_recall"].mean(),

        basic_results2["test_recall"].mean(),

        pruned_results2["test_recall"].mean(),

        rf_results2["test_recall"].mean()
    ],

    "Precision":[

        dummy_results2["test_precision"].mean(),

        basic_results2["test_precision"].mean(),

        pruned_results2["test_precision"].mean(),

        rf_results2["test_precision"].mean()
    ],

    "F1 Score":[

        dummy_results2["test_f1"].mean(),

        basic_results2["test_f1"].mean(),

        pruned_results2["test_f1"].mean(),

        rf_results2["test_f1"].mean()
    ],

    "ROC AUC":[

        dummy_results2["test_roc_auc"].mean(),

        basic_results2["test_roc_auc"].mean(),

        pruned_results2["test_roc_auc"].mean(),

        rf_results2["test_roc_auc"].mean()
    ],

    "PR AUC":[

        dummy_results2["test_pr_auc"].mean(),

        basic_results2["test_pr_auc"].mean(),

        pruned_results2["test_pr_auc"].mean(),

        rf_results2["test_pr_auc"].mean()

    ]

})

comparison2 = comparison2.sort_values(
    by="ROC AUC",
    ascending=False
)

display(comparison2)

# Visual Comparison

plt.figure(figsize=(10,6))

sns.barplot(
    data=comparison2,
    x="Model",
    y="ROC AUC",
    palette="viridis"
)

plt.title("Heart Disease Model Comparison")

plt.xticks(rotation=15)

plt.show()
