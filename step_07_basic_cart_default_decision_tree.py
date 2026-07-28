"""
Step 7: Basic CART (Default Decision Tree)
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #7
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Basic CART (Default Decision Tree)

# Basic CART Model

basic_cart = DecisionTreeClassifier(
    criterion="gini",
    random_state=RANDOM_STATE
)

basic_cv = cross_validate(
    basic_cart,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True
)

basic_results = pd.DataFrame(basic_cv)

summary_basic = pd.DataFrame({
    "Train Mean": basic_results.filter(regex="train").mean(),
    "Test Mean": basic_results.filter(regex="test").mean(),
    "Train Std": basic_results.filter(regex="train").std(),
    "Test Std": basic_results.filter(regex="test").std()
})

display(summary_basic)

# Fit Basic CART

basic_cart.fit(X_train, y_train)

print("Tree Depth :", basic_cart.get_depth())
print("Number of Leaves :", basic_cart.get_n_leaves())
print("Number of Nodes :", basic_cart.tree_.node_count)
