"""
Step 34: Hyperparameter Tuning
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #34
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Hyperparameter Tuning

tree_pipe2 = Pipeline([
    ("preprocessor",preprocessor),
    ("model",DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ))
])

# Parameter Grid

param_grid2={

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
        1,2,5,10
    ],

    "model__class_weight":[
        None,
        "balanced"
    ]
}

# Grid Search

grid2=GridSearchCV(

    estimator=tree_pipe2,

    param_grid=param_grid2,

    scoring=scoring,

    refit="roc_auc",

    cv=cv2,

    n_jobs=-1,

    return_train_score=True
)

grid2.fit(
    X2_train,
    y2_train
)

print(grid2.best_params_)

print("\nBest ROC AUC")

print(grid2.best_score_)
