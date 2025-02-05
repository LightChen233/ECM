from utils.tools import read_jsonl



import matplotlib.pyplot as plt
import numpy as np



mu = 200
sigma = 25
n_bins = 25
data = np.array([flux["flux"] for d in read_jsonl("experiments/icl-explanation/synthetic-cot/data/test_pos.jsonl") for flux in d["demo_list"][:]] + [flux["flux"] for d in read_jsonl("experiments/icl-explanation/synthetic-cot/data/test_neg.jsonl") for flux in d["demo_list"][:]])

fig = plt.figure(figsize=(9, 4), layout="constrained")
axs = fig.subplots(1, 2, sharex=True, sharey=True)

# Cumulative distributions.
axs[0].ecdf(data, label="CDF")
n, bins, patches = axs[0].hist(data, n_bins, density=True, histtype="step",
                               cumulative=True)

x = np.linspace(data.min(), data.max())
y = np.array([0.5 for _ in x])
axs[0].plot(x, y, "k--", linewidth=1.5)
# Complementary cumulative distributions.
axs[1].ecdf(data, complementary=True, label="CCDF")
axs[1].hist(data, bins=bins, density=True, histtype="step", cumulative=-1,)
axs[1].plot(x, 1 - y, "k--", linewidth=1.5)

# Label the figure.
for ax in axs:
    ax.grid(True)
    ax.legend()
    ax.label_outer()
plt.savefig('experiments/icl-explanation/synthetic-cot/result-flux.svg', format='svg')
plt.show()