"""
Step 63: Summary Statistics
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #63
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Summary Statistics

summary_stats = stability_df.describe().T[
    ["mean", "std", "min", "max"]
]

display(summary_stats)
