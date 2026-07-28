"""
Step 22: Import Additional Libraries
Source notebook : 23MID0121_Lab02_DecisionTree.ipynb
Original cell   : #22
"""

from pipeline_shared import *  # noqa: F401,F403 (shared imports / variables from earlier steps)

# Import Additional Libraries

from sklearn.datasets import fetch_openml

from sklearn.compose import make_column_selector

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
