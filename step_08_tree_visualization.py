"""
Step 8: Tree Visualization
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #8
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Tree Visualization

plt.figure(figsize=(22,10))

plot_tree(
    basic_cart,
    feature_names=X.columns,
    class_names=["Benign","Malignant"],
    filled=True,
    rounded=True,
    proportion=True,
    max_depth=3,
    fontsize=10
)

plt.title("Basic CART (Top 3 Levels)")

plt.show()

# Display Decision Rules

rules = export_text(
    basic_cart,
    feature_names=list(X.columns)
)

print(rules[:5000])
