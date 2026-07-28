"""
Step 44: Feature Importance
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #44
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Feature Importance

# Impurity-Based Importance

final_tree2 = final_model2.named_steps["model"]

feature_names = final_model2.named_steps[
    "preprocessor"
].get_feature_names_out()

importance2 = pd.Series(

    final_tree2.feature_importances_,

    index=feature_names

).sort_values(
    ascending=False
)

display(
    importance2.head(15)
)

# Plot Feature Importance

plt.figure(figsize=(8,6))

importance2.head(10).plot(
    kind="barh"
)

plt.gca().invert_yaxis()

plt.title("Top 10 Important Features")

plt.show()

# Permutation Importance

perm2 = permutation_importance(

    final_model2,

    X2_test,

    y2_test,

    scoring="roc_auc",

    n_repeats=30,

    random_state=RANDOM_STATE
)

perm_df2 = pd.DataFrame({

    "Feature": X2.columns,

    "Importance": perm2.importances_mean,

    "Std": perm2.importances_std

}).sort_values(
    by="Importance",
    ascending=False
)

display(
    perm_df2.head(15)
)
