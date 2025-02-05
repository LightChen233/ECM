import random
random.seed(42)
from utils.tools import read_jsonl, write_jsonl

train_data_path = fr"data/gsm8k/train-skill_embed.jsonl"
train_data = read_jsonl(train_data_path)
train_matrix = []
SHOT = 5
golden_group = random.sample(list(range(len(train_data))), k = SHOT)

print(golden_group)
res_data = []
for x in golden_group:
    train_data[x]["index"] = str(x)
    res_data.append(train_data[x])
write_jsonl("experiments/icl_exploration/diversity/data/random-icl/gsm8k_train_diverse_sample", res_data, "w")