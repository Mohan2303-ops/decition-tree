"""
Step 19: Feature Importance
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #19
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Feature Importance

# Impurity Importance

fitted_tree = final_model.named_steps["model"]

importance = pd.Series(

    fitted_tree.feature_importances_,

    index=X.columns

).sort_values(
    ascending=False
)

display(
    importance.head(15)
)

# Plot Feature Importance

plt.figure(figsize=(8,6))

importance.head(10).plot(
    kind="barh"
)

plt.title("Top 10 Feature Importance")

plt.gca().invert_yaxis()

plt.show()

# Permutation Importance

perm = permutation_importance(

    final_model,

    X_test,

    y_test,

    scoring="roc_auc",

    n_repeats=30,

    random_state=RANDOM_STATE
)

perm_df = pd.DataFrame({

    "Feature":X.columns,

    "Importance":perm.importances_mean,

    "Std":perm.importances_std

}).sort_values(
    by="Importance",
    ascending=False
)

display(
    perm_df.head(15)
)
