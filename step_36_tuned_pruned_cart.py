"""
Step 36: Tuned & Pruned CART
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #36
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Tuned & Pruned CART

pruned_cv2 = cross_validate(

    best_tree2,

    X2_train,

    y2_train,

    cv=cv2,

    scoring=scoring,

    return_train_score=True
)

pruned_results2 = pd.DataFrame(
    pruned_cv2
)

summary_pruned2 = pd.DataFrame({

    "Train Mean":pruned_results2.filter(regex="train").mean(),

    "Test Mean":pruned_results2.filter(regex="test").mean(),

    "Train Std":pruned_results2.filter(regex="train").std(),

    "Test Std":pruned_results2.filter(regex="test").std()

})

display(summary_pruned2)

# Visualize Final Tree

final_tree2 = best_tree2.named_steps["model"]

plt.figure(figsize=(22,10))

plot_tree(

    final_tree2,

    feature_names=feature_names,

    class_names=[
        "No Disease",
        "Disease"
    ],

    filled=True,

    rounded=True,

    proportion=True

)

plt.title("Final Tuned & Pruned CART")

plt.show()
