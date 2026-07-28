"""
Step 29: EDA
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #29
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# EDA

# Numerical Features

for feature in numerical_columns:

    plt.figure(figsize=(7,5))

    sns.histplot(
        data=pd.concat([X2, y2], axis=1),
        x=feature,
        hue="HeartDisease",
        kde=True,
        stat="density",
        common_norm=False
    )

    plt.show()

# Categorical Features

eda2 = X2.copy()
eda2["HeartDisease"] = y2

for feature in categorical_columns:

    plt.figure(figsize=(7,4))

    sns.countplot(
        data=eda2,
        x=feature,
        hue="HeartDisease"
    )

    plt.xticks(rotation=30)

    plt.show()
