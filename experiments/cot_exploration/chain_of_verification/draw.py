from matplotlib import pyplot as plt
import pandas as pd
# import scipy
# stat, pvalue = scipy.stats.ttest_ind(a, b)

import seaborn as sns

# from statannotations.Annotator import Annotator

df = pd.DataFrame([
    {"key": 0, "value": 15.15},
    {"key": 0, "value": 13.94},
    {"key": 0, "value": 12.73},
    {"key": 0, "value": 11.52},
    {"key": 0, "value": 11.52},
    {"key": 1, "value": 21.81},
    {"key": 1, "value": 18.18},
    {"key": 1, "value": 15.75},
    {"key": 1, "value": 16.36},
    {"key": 1, "value": 15.15},
    {"key": 2, "value": 31.52},
    {"key": 2, "value": 29.09},
    {"key": 2, "value": 30.91},
    {"key": 2, "value": 30.91},
    {"key": 2, "value": 31.52},
    {"key": 3, "value": 27.27},
    {"key": 3, "value": 27.88},
    {"key": 3, "value": 28.48},
    {"key": 3, "value": 29.09},
    {"key": 3, "value": 25.45},
    {"key": 4, "value": 23.64},
    {"key": 4, "value": 25.45},
    {"key": 4, "value": 25.45},
    {"key": 4, "value": 21.82},
    {"key": 4, "value": 22.42},
    {"key": 5, "value": 17.58},
    {"key": 5, "value": 19.39},
    {"key": 5, "value": 21.82},
    {"key": 5, "value": 21.21},
    {"key": 5, "value": 19.39},
    ])
x = "key"
y = "value"

sns.catplot(
    data=df, x=x, y=y,
    capsize=.2, palette="YlGnBu_d", errorbar="se", 
    kind="point", height=6, aspect=.75
)

sns.stripplot(data=df, x=x, y=y, size=4, color=".3")
plt.savefig("experiments/cot_exploration/chain_of_verification/result.svg", format="svg")
plt.show()