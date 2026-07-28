"""
Step 20: Decision Path Interpretation
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #20
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Decision Path Interpretation

# Display Final Tree

plt.figure(figsize=(22,10))

plot_tree(

    fitted_tree,

    feature_names=X.columns,

    class_names=[
        "Benign",
        "Malignant"
    ],

    filled=True,

    rounded=True,

    proportion=True,

    fontsize=10

)

plt.title("Final Tuned & Pruned CART")

plt.show()

# Print Rules

rules = export_text(

    fitted_tree,

    feature_names=list(X.columns)

)

print(rules)
