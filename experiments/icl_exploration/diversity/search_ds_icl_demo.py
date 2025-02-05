import os
import random
random.seed(42)
import torch
from tqdm import tqdm
from utils.tools import read_jsonl, write_jsonl


train_data_path = f"semantic/skill/gsm8k_train.jsonl"
train_data = [torch.tensor(td["embed"]) for td in read_jsonl(train_data_path)]
train_matrix = []
temp_file_path = "experiments/icl_exploration/diversity/data/ds-icl/diversity_matrix.jsonl"
save_path = "experiments/icl_exploration/diversity/data/ds-icl/gsm8k_train_diverse_sample.jsonl"
if os.path.exists(temp_file_path):
    last_idx = [i for i, x in enumerate(read_jsonl(temp_file_path))]
    train_matrix = read_jsonl(temp_file_path)
else:
    last_idx = []
for i, td in enumerate(tqdm(train_data)):
    if i in last_idx:
        continue
    train_matrix.append([])
    train_embed = td
    for j, td_2 in enumerate(train_data):
        value = 1
        if i < len(train_matrix) and j < len(train_matrix[i]):
            value = train_matrix[i][j]
        elif i != j:
            train_embed_2 = td_2
            value = torch.cosine_similarity(train_embed, train_embed_2, dim=0).item()
        train_matrix[i].append(value)
    write_jsonl(temp_file_path, [train_matrix[i]], "a")
SHOT = 5
golden_group = []
golden_group_value = 1000000
sample_list = [random.sample(list(range(len(train_data))), k = SHOT) for _ in range(10000)]

for group in tqdm(sample_list):
    group_value = 0
    for i in group:
        for j in group:
            if i != j:
                group_value += train_matrix[i][j]
    if golden_group_value > group_value:
        golden_group_value = group_value
        golden_group = group


print(golden_group)
train_data = read_jsonl(train_data_path)
res_data = []
for x in golden_group:
    train_data[x]["index"] = str(x)
    res_data.append(train_data[x])
write_jsonl(save_path, res_data, "w")