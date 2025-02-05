import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()

data_path = "experiments/explanation/request_data/ICL_Linear(in_features=4096, out_features=4096, bias=False)_attention.csv"

input_data = []
with open(data_path, "r", encoding="utf8") as f:
    for line in f:
        input_data.append({"value": float(line.strip())})
flights = pd.DataFrame(input_data)

f, ax = plt.subplots(figsize=(1, 9)) # figsize=(9, 6)
sns.heatmap(flights, annot=False, linewidths=0, ax=ax)
plt.savefig('experiments/explanation/icl.svg', format='svg')
plt.show()