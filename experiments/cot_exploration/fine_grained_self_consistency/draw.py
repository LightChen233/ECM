from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame([
    {"key": 0, "value": 52},
    {"key": 0, "value": 52},
    {"key": 0, "value": 52},
    {"key": 1, "value": 56.56},
    {"key": 1, "value": 58.52},
    {"key": 1, "value": 60.49},
    {"key": 1, "value": 58.20},
    {"key": 1, "value": 57.54},
    {"key": 2, "value": 58.36},
    {"key": 2, "value": 59.51},
    {"key": 2, "value": 60.49},
    {"key": 2, "value": 58.20},
    {"key": 2, "value": 56.72},
    {"key": 3, "value": 64.43},
    {"key": 3, "value": 59.67},
    {"key": 3, "value": 59.84},
    {"key": 3, "value": 61.15},
    {"key": 3, "value": 62.62},
    {"key": 4, "value": 62.79},
    {"key": 4, "value": 63.28},
    {"key": 4, "value": 59.51},
    {"key": 4, "value": 60.66},
    {"key": 4, "value": 60.49},
    ])
x = "key"
y = "value"
ax = sns.lineplot(data=df, x=x, y=y)
plt.savefig('experiments/ICL/fine_grained_self_consistency/result.svg', format='svg')
plt.show()