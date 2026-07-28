"""
Step 57: Discussion (Markdown)
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #57
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Discussion (Markdown)

# Comparison of Breast Cancer and Heart Disease Datasets

# • The Breast Cancer dataset contains only numerical features and no missing values, making it easier to model.

# • The Heart Disease dataset contains both numerical and categorical variables along with missing values, requiring preprocessing before training.

# • Random Forest achieved the highest overall performance on both datasets due to its ensemble learning capability.

# • The Tuned & Pruned CART model consistently outperformed the Basic CART model, indicating that hyperparameter tuning and pruning improved generalization.

# • The Dummy classifier performed the worst on both datasets, confirming that the tree-based models learned meaningful patterns.

# • Although preprocessing was more complex for the Heart Disease dataset, the machine learning pipeline successfully handled missing values and categorical variables.

# • Using the same evaluation protocol for both datasets enables a fair comparison of model performance.
