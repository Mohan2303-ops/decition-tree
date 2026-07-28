"""
Step 32: Basic CART
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #32
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Basic CART

# Build Basic CART Pipeline

basic_cart2 = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        criterion="gini",
        random_state=RANDOM_STATE
    ))
])

# Cross Validation

basic_cv2 = cross_validate(

    basic_cart2,

    X2_train,

    y2_train,

    cv=cv2,

    scoring=scoring,

    return_train_score=True
)

basic_results2 = pd.DataFrame(basic_cv2)

summary_basic2 = pd.DataFrame({

    "Train Mean":basic_results2.filter(regex="train").mean(),

    "Test Mean":basic_results2.filter(regex="test").mean(),

    "Train Std":basic_results2.filter(regex="train").std(),

    "Test Std":basic_results2.filter(regex="test").std()

})

display(summary_basic2)

# Fit Model

basic_cart2.fit(
    X2_train,
    y2_train
)

tree = basic_cart2.named_steps["model"]

print("Tree Depth :",tree.get_depth())

print("Leaves :",tree.get_n_leaves())

print("Nodes :",tree.tree_.node_count)
