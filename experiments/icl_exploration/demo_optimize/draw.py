from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns

from utils.tools import read_jsonl
sns.set_theme(style="whitegrid")
res = read_jsonl("experiments/icl_exploration/demo_optimize/request_data/final.jsonl")
data = {}
for d in res:
    for key in d:
        if key not in data:
            data[key] = []
        data[key].append(d[key])
    

df = DataFrame(data)

g = sns.catplot(
    data=df, x="iter", y="acc", hue="type",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.savefig("experiments/icl_exploration/demo_optimize/result.svg", format='svg')
plt.show()
data = {}
for d in res:
    if d["temperature"] == "05":
        for key in d:
            if key not in data:
                data[key] = []
            data[key].append(d[key])
    

df = DataFrame(data)

g = sns.catplot(
    data=df, x="iter", y="p", hue="type",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.savefig("experiments/icl_exploration/demo_optimize/result_p.svg", format='svg')
plt.show()