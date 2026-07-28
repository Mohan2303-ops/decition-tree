"""
Step 33: Tree Visualization
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #33
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Tree Visualization

feature_names = basic_cart2.named_steps[
    "preprocessor"
].get_feature_names_out()

plt.figure(figsize=(22,10))

plot_tree(

    tree,

    feature_names=feature_names,

    class_names=[
        "No Disease",
        "Disease"
    ],

    filled=True,

    rounded=True,

    proportion=True,

    max_depth=3

)

plt.title("Basic CART")

plt.show()

#Decision Rules

rules = export_text(
    tree,
    feature_names=list(feature_names)
)

print(rules[:5000])
