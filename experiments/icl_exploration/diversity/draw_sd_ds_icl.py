from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns

from utils.tools import read_jsonl
sns.set_theme(style="whitegrid")

data = {
    "acc": [0.5908346972176759, 0.5782537067545305, 0.5515548281505729, 0.5311475409836065, 0.5377049180327869, 0.5327868852459017, 0.4852459016393443, 0.5262295081967213, 0.5311475409836065, 972.1385573379819 / 1000 * 0.2 + 0.38, 714.2512014686812  / 1000 * 0.2 + 0.38, 686.3397982834572 / 1000 * 0.2 + 0.38, ],
    "key": ["acc", "acc", "acc", "acc", "acc", "acc", "acc", "acc", "acc", "p", "p", "p"],
    "type": ["sd-icl", "sd-icl", "sd-icl", "ds-icl", "ds-icl", "ds-icl", "random-icl", "random-icl", "random-icl", "sd-icl", "ds-icl", "random-icl"]
}
df = DataFrame(data)

g = sns.barplot(
    data=df, x="type", y="acc", hue="key",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
)
g.set_ylim(0.4,0.6)
plt.savefig("experiments/icl_exploration/diversity/sd_ds_icl_result.svg", format='svg')
plt.show()
