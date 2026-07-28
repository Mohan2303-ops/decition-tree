"""
Step 35: Cost Complexity Pruning
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #35
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Cost Complexity Pruning

# Alpha Values

X2_processed = preprocessor.fit_transform(
    X2_train
)

base_tree2 = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)

path2 = base_tree2.cost_complexity_pruning_path(
    X2_processed,
    y2_train
)

ccp_alphas2=np.unique(
    path2.ccp_alphas[:-1]
)

print(len(ccp_alphas2))

# Alpha Search

alpha_grid2={

    "model__ccp_alpha":ccp_alphas2,

    "model__class_weight":[
        None,
        "balanced"
    ]
}

prune_search2=GridSearchCV(

    tree_pipe2,

    alpha_grid2,

    scoring=scoring,

    refit="roc_auc",

    cv=cv2,

    n_jobs=-1,

    return_train_score=True
)

prune_search2.fit(
    X2_train,
    y2_train
)

print(prune_search2.best_params_)

print(prune_search2.best_score_)

# Final Tuned Tree

best_tree2 = prune_search2.best_estimator_

best_tree2.fit(
    X2_train,
    y2_train
)
