"""
Step 10: Cost Complexity Pruning
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #10
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Cost Complexity Pruning

# Find Alpha Values

base_tree = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)

path = base_tree.cost_complexity_pruning_path(
    X_train,
    y_train
)

ccp_alphas = np.unique(path.ccp_alphas[:-1])

print("Number of Alpha Values :",len(ccp_alphas))

# Grid Search for Best Alpha

alpha_grid = {

    "model__ccp_alpha":ccp_alphas,

    "model__class_weight":[
        None,
        "balanced"
    ]
}

prune_search = GridSearchCV(

    tree_pipe,

    alpha_grid,

    scoring=scoring,

    refit="roc_auc",

    cv=cv,

    n_jobs=-1,

    return_train_score=True
)

prune_search.fit(
    X_train,
    y_train
)

print("Best Alpha")

print(prune_search.best_params_)

print("\nBest ROC AUC")

print(prune_search.best_score_)

# Best Tuned Model

best_tree = prune_search.best_estimator_

best_tree.fit(
    X_train,
    y_train
)
