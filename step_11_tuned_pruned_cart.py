"""
Step 11: Tuned & Pruned CART
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #11
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Tuned & Pruned CART

pruned_cv = cross_validate(

    best_tree,

    X_train,

    y_train,

    cv=cv,

    scoring=scoring,

    return_train_score=True
)

pruned_results = pd.DataFrame(pruned_cv)

summary_pruned = pd.DataFrame({

    "Train Mean":pruned_results.filter(regex="train").mean(),

    "Test Mean":pruned_results.filter(regex="test").mean(),

    "Train Std":pruned_results.filter(regex="train").std(),

    "Test Std":pruned_results.filter(regex="test").std()

})

display(summary_pruned)

# Visualize Final Pruned Tree

fitted_tree = best_tree.named_steps["model"]

plt.figure(figsize=(20,10))

plot_tree(
    fitted_tree,
    feature_names=X.columns,
    class_names=["Benign","Malignant"],
    filled=True,
    rounded=True,
    proportion=True
)

plt.title("Final Tuned & Pruned CART")

plt.show()
