"""
Step 23: Extract target column
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #23
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

heart = fetch_openml(
    name="heart-disease",
    version=1,
    as_frame=True
)

X2 = heart.data.copy()

# Extract target column
y2 = X2.pop("target")

display(X2.head())
display(y2.head())

print(y2.unique())
print(y2.value_counts())
print(y2.dtype)
