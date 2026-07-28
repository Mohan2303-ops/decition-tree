"""
Step 14: Threshold Experiment
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #14
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Threshold Experiment

# Generate Out-of-Fold Probabilities

selected_model = prune_search.best_estimator_

oof_prob = cross_val_predict(
    selected_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:,1]

# Find Best Threshold

threshold_results = []

for threshold in np.linspace(0.05,0.95,181):

    prediction = (oof_prob >= threshold).astype(int)

    tn,fp,fn,tp = confusion_matrix(
        y_train,
        prediction
    ).ravel()

    sensitivity = tp/(tp+fn) if(tp+fn)!=0 else 0

    specificity = tn/(tn+fp) if(tn+fp)!=0 else 0

    threshold_results.append([
        threshold,
        sensitivity,
        specificity
    ])

threshold_df = pd.DataFrame(
    threshold_results,
    columns=[
        "Threshold",
        "Sensitivity",
        "Specificity"
    ]
)

display(threshold_df.head())

# Select Threshold

TARGET_SENSITIVITY = 0.90

candidate = threshold_df[
    threshold_df["Sensitivity"]>=TARGET_SENSITIVITY
]

selected_threshold = candidate.sort_values(
    by="Specificity",
    ascending=False
).iloc[0]["Threshold"]

print("Selected Threshold =",selected_threshold)

# Threshold Plot

plt.figure(figsize=(10,5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Sensitivity"],
    label="Sensitivity"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Specificity"],
    label="Specificity"
)

plt.axvline(
    selected_threshold,
    color="red",
    linestyle="--",
    label="Selected Threshold"
)

plt.legend()

plt.xlabel("Threshold")

plt.ylabel("Score")

plt.title("Threshold Selection")

plt.show()
