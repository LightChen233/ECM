from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns

from utils.tools import read_jsonl
sns.set_theme(style="whitegrid")
res = read_jsonl("experiments/icl_exploration/diversity/request_data/dd-icl/final.jsonl")
data = {}
for d in res:
    for key in d:
        if key not in data:
            data[key] = []
        data[key].append(d[key])

df = DataFrame(data)

g = sns.catplot(
    data=df, x="top", y="acc", hue="key",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.savefig("experiments/icl_exploration/diversity/dd_icl_result.svg", format='svg')
plt.show()
