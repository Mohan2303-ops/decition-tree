"""
Step 39: Threshold Experiment
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #39
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Threshold Experiment

# Generate Out-of-Fold Probabilities

selected_model2 = prune_search2.best_estimator_

oof_prob2 = cross_val_predict(
    selected_model2,
    X2_train,
    y2_train,
    cv=cv2,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# Evaluate Different Thresholds

threshold_results2 = []

for threshold in np.linspace(0.05, 0.95, 181):

    prediction = (oof_prob2 >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y2_train,
        prediction
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    threshold_results2.append([
        threshold,
        sensitivity,
        specificity
    ])

threshold_df2 = pd.DataFrame(
    threshold_results2,
    columns=[
        "Threshold",
        "Sensitivity",
        "Specificity"
    ]
)

display(threshold_df2.head())

# Select Best Threshold

TARGET_SENSITIVITY = 0.90

candidate = threshold_df2[
    threshold_df2["Sensitivity"] >= TARGET_SENSITIVITY
]

selected_threshold2 = candidate.sort_values(
    by="Specificity",
    ascending=False
).iloc[0]["Threshold"]

print("Selected Threshold:", selected_threshold2)

# Threshold Plot

plt.figure(figsize=(10,5))

plt.plot(
    threshold_df2["Threshold"],
    threshold_df2["Sensitivity"],
    label="Sensitivity"
)

plt.plot(
    threshold_df2["Threshold"],
    threshold_df2["Specificity"],
    label="Specificity"
)

plt.axvline(
    selected_threshold2,
    color="red",
    linestyle="--",
    label="Selected Threshold"
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Selection")

plt.legend()

plt.show()
