"""
Step 5: Train-Test Split
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #5
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print("\nTraining Class Distribution")

print(y_train.value_counts())

print("\nTesting Class Distribution")

print(y_test.value_counts())
