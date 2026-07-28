"""
Step 4: Exploratory Data Analysis (EDA)
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #4
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Exploratory Data Analysis (EDA)

# 1 Class Distribution

plt.figure(figsize=(6,5))

sns.countplot(
    x=y,
    palette="Set2"
)

plt.xticks(
    [0,1],
    ["Benign","Malignant"]
)

plt.title("Class Distribution")

plt.xlabel("Class")

plt.ylabel("Count")

plt.show()

# 2 Missing Value Heatmap

plt.figure(figsize=(12,4))

sns.heatmap(
    X.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values")

plt.show()

# 3 Selected Feature Distribution

selected_features = [
    "mean radius",
    "mean texture",
    "mean concavity",
    "worst radius"
]

eda = X[selected_features].copy()

eda["Class"] = y

eda["Class"] = eda["Class"].map({
    0:"Benign",
    1:"Malignant"
})

for feature in selected_features:

    plt.figure(figsize=(7,5))

    sns.histplot(
        data=eda,
        x=feature,
        hue="Class",
        kde=True,
        stat="density",
        common_norm=False
    )

    plt.title(feature)

    plt.show()


# 4 Boxplots

for feature in selected_features:

    plt.figure(figsize=(7,5))

    sns.boxplot(
        x="Class",
        y=feature,
        data=eda
    )

    plt.title(feature)

    plt.show()

# 5 Correlation Heatmap

plt.figure(figsize=(8,6))

sns.heatmap(
    X[selected_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# EDA Observations

# 1. The dataset contains 569 observations and 30 numerical features.

# 2. There are no missing values.

# 3. The malignant class is smaller than the benign class, making stratified sampling appropriate.

# 4. Features such as mean radius, worst radius, and mean concavity show good separation between malignant and benign cases.

# 5. Some numerical predictors are strongly correlated, which is acceptable for tree-based models but should be noted.
