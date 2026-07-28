"""
Step 68: Model Card
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #68
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Model Card

# Model Information

model_card = {
    "Model Name": "Random Forest Classifier",
    "Project": "Comparative Study of Tree-Based Models for Medical Diagnosis",
    "Version": "1.0",
    "Author": "Your Name",
    "Date": pd.Timestamp.today().strftime("%Y-%m-%d"),
    "Framework": "Scikit-learn"
}

pd.DataFrame(
    model_card.items(),
    columns=["Field", "Value"]
)

# Dataset Information

dataset_info = pd.DataFrame({

    "Dataset":[
        "Breast Cancer Wisconsin",
        "UCI Heart Disease"
    ],

    "Samples":[
        len(X),
        len(X2)
    ],

    "Features":[
        X.shape[1],
        X2.shape[1]
    ],

    "Target":[
        "Malignant",
        "Heart Disease"
    ],

    "Missing Values":[
        "No",
        "Yes"
    ],

    "Feature Types":[
        "Numerical",
        "Numerical + Categorical"
    ]
})

display(dataset_info)

# Model Performance Summary

performance = pd.DataFrame({

    "Dataset":[
        "Breast Cancer",
        "Heart Disease"
    ],

    "Best Model":[
        comparison.iloc[0]["Model"],
        comparison2.iloc[0]["Model"]
    ],

    "ROC AUC":[
        comparison.iloc[0]["ROC AUC"],
        comparison2.iloc[0]["ROC AUC"]
    ],

    "F1 Score":[
        comparison.iloc[0]["F1 Score"],
        comparison2.iloc[0]["F1 Score"]
    ],

    "Balanced Accuracy":[
        comparison.iloc[0]["Balanced Accuracy"],
        comparison2.iloc[0]["Balanced Accuracy"]
    ]

})

display(performance)

# Intended Use

# This model is intended for educational purposes to demonstrate the application of Decision Tree and Random Forest algorithms for binary medical classification tasks.

# It may be used to compare different tree-based learning methods and preprocessing strategies on benchmark medical datasets.

# Out-of-Scope Use

# • This model should not be used for clinical diagnosis.

# • It should not replace healthcare professionals.

# • It has not been externally validated on real hospital data.

# • Predictions must not be interpreted as medical advice.

# Training Details

training = pd.DataFrame({

    "Parameter":[

        "Train-Test Split",

        "Cross Validation",

        "Random State",

        "Evaluation Metric",

        "Threshold Selection"

    ],

    "Value":[

        "80 : 20",

        "5-Fold Stratified",

        RANDOM_STATE,

        "ROC-AUC",

        "Training Data Only"

    ]

})

display(training)

# Limitations

# • The datasets are relatively small.

# • External validation was not performed.

# • Class distributions may differ from real clinical populations.

# • Performance may decrease on unseen hospital data.

# • Decision Trees are sensitive to the training data, although pruning reduces overfitting.

# • Random Forest models are less interpretable than a single Decision Tree.

# Fairness & Ethical Considerations

# • The datasets may not represent all patient populations.

# • Model predictions could vary across demographic groups.

# • Fairness metrics were not explicitly evaluated.

# • Human oversight is required before any healthcare decision.

# Reproducibility

reproducibility = pd.DataFrame({

    "Item":[

        "Python",

        "Scikit-learn",

        "Random State",

        "Cross Validation",

        "Evaluation"

    ],

    "Value":[

        "3.x",

        "Latest Stable Version",

        RANDOM_STATE,

        "5-Fold Stratified",

        "ROC-AUC, PR-AUC, F1, Balanced Accuracy"

    ]

})

display(reproducibility)

# Model Card Summary

# Model:
# Random Forest Classifier

# Datasets:
# • Breast Cancer Wisconsin (Diagnostic)
# • UCI Heart Disease (Cleveland)

# Task:
# Binary Medical Classification

# Evaluation:
# • Accuracy
# • Balanced Accuracy
# • Precision
# • Recall
# • F1 Score
# • ROC-AUC
# • PR-AUC

# Validation:
# 5-Fold Stratified Cross Validation

# Strengths:
# • High predictive performance
# • Handles nonlinear relationships
# • Robust to noisy data
# • Stable across different train-test splits

# Limitations:
# • Not clinically validated
# • Limited dataset size
# • Educational use only

# Ethics:
# Human oversight is mandatory. The model should support, not replace, professional medical judgment.
