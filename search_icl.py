import torch
from tqdm import tqdm
from utils.tools import read_jsonl, write_jsonl
import fire

def run(
        setting="semantic", # ["skill", "semantic"]
        model="roberta", # ["skill", "semantic"]
        data_name="biggsm", # ["biggsm", "gsm8k"]
        model_name="gpt35", # ["gpt4", "gpt35"]
    ):
        if setting == "semantic":
            test_data_path = fr"semantic/{model}_embed/{data_name}_test.jsonl"
        else:
            test_data_path = fr"semantic/skill/{data_name}/test-{model_name}-tp-01_embed-1.jsonl"
        test_data = read_jsonl(test_data_path)
        if data_name == "biggsm":
            data_name = "gsm8k"
        if setting == "semantic":
            train_data_path = fr"semantic/{model}_embed/{data_name}_train.jsonl"
        else:
            train_data_path = fr"semantic/skill/{data_name}/train-{model_name}-tp-01_embed.jsonl"
        train_data = read_jsonl(train_data_path)
        last_idx = [x["index"] for x in test_data if "demo_list" in x]
        for i, test_d in enumerate(tqdm(test_data)):
            # if test_d["index"] in last_idx:
            #     continue
            idx_list = []
            test_embed = torch.tensor(test_d["embed"])
            norm_test = torch.norm(test_embed)
            for t_i, train_d in enumerate(train_data):
                train_embed = torch.tensor(train_d["embed"])
                flux = torch.dot(test_embed, train_embed) / norm_test
                if "index" in train_d:
                    idx_list.append({"index": train_d["index"], "flux": flux.item()})
                else:
                    idx_list.append({"index": str(t_i), "flux": flux.item()})
            idx_list = sorted(idx_list, key=lambda x: x["flux"], reverse=True)
            if len(idx_list) == 0:
                print(1)
            test_data[i]["demo_list"] = idx_list
        write_jsonl(test_data_path, test_data, "w")

if __name__ == "__main__":
    fire.Fire(run)