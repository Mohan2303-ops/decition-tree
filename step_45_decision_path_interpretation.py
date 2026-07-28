"""
Step 45: Decision Path Interpretation
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #45
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Decision Path Interpretation

# Visualize Final Tree

plt.figure(figsize=(22,10))

plot_tree(

    final_tree2,

    feature_names=feature_names,

    class_names=[
        "No Disease",
        "Disease"
    ],

    filled=True,

    rounded=True,

    proportion=True,

    fontsize=10

)

plt.title("Final Tuned & Pruned CART")

plt.show()

# Decision Rules

rules2 = export_text(

    final_tree2,

    feature_names=list(feature_names)

)

print(rules2)
