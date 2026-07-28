"""
Step 31: Preprocessing Pipeline + Dummy Baseline
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #31
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Preprocessing Pipeline + Dummy Baseline

# Identify Feature Types

categorical_columns = X2_train.select_dtypes(
    include=["object","category"]
).columns.tolist()

numerical_columns = X2_train.select_dtypes(
    include=["int64","float64"]
).columns.tolist()

print("Categorical Columns")

print(categorical_columns)

print("\nNumerical Columns")

print(numerical_columns)

# Build Preprocessor

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_columns),
    ("cat", categorical_transformer, categorical_columns)
])


# Dummy Baseline

dummy_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyClassifier(strategy="prior"))
])

dummy_cv2 = cross_validate(
    dummy_pipeline,
    X2_train,
    y2_train,
    cv=cv2,
    scoring=scoring,
    return_train_score=False
)

dummy_results2 = pd.DataFrame(dummy_cv2).filter(regex="test_")

summary_dummy2 = pd.DataFrame({
    "Mean": dummy_results2.mean(),
    "Std": dummy_results2.std()
})

display(summary_dummy2)
