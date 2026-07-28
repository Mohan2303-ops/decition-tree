"""
Step 9: Hyperparameter Tuning
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #9
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Hyperparameter Tuning

# Create Pipeline

tree_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
])

# Parameter Grid

param_grid = {

    "model__criterion":[
        "gini",
        "entropy",
        "log_loss"
    ],

    "model__max_depth":[
        2,3,4,5,6,8,None
    ],

    "model__min_samples_split":[
        2,5,10,20
    ],

    "model__min_samples_leaf":[
        1,2,5,10,20
    ],

    "model__class_weight":[
        None,
        "balanced"
    ]
}

# Grid Search

grid = GridSearchCV(

    estimator=tree_pipe,

    param_grid=param_grid,

    scoring=scoring,

    refit="roc_auc",

    cv=cv,

    n_jobs=-1,

    return_train_score=True
)

grid.fit(X_train,y_train)

print("Best Parameters\n")

print(grid.best_params_)

print("\nBest ROC AUC")

print(grid.best_score_)
