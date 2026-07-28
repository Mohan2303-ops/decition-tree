"""
Step 30: Train-Test Split
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #30
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Train-Test Split

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y2,
    test_size=0.20,
    stratify=y2,
    random_state=RANDOM_STATE
)

cv2 = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

print("Training Shape :", X2_train.shape)
print("Testing Shape :", X2_test.shape)

print("\nTraining Distribution")

print(y2_train.value_counts())

print("\nTesting Distribution")

print(y2_test.value_counts())
