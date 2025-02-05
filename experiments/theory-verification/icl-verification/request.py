import asyncio
from functools import partial
import os
import random
import fire

from utils.request_tool import request_LLM
from utils.tools import read_jsonl

random.seed(42)


def create_prompt(data, prompt_config):
    instruction = f""""""
    for d in data["request_list"]:
        instruction += f"""[QUESTION]
{d['question']}
[ANSWER]
{d['answer']}

"""
    instruction += f"""[QUESTION]
{data['question']}"""
    return instruction


class DataLoader:
    def __init__(self, test_data_path, train_data, shot) -> None:
        input_data = []
        for i, data in enumerate(read_jsonl(test_data_path)):
            if "index" not in data:
                data["index"] = str(i)
            if "embed" in data:
                del data["embed"]
            data["request_list"] = []
            for d in data["demo_list"][:shot]:
                data["request_list"].append(train_data[int(d["index"])])
            data["demo_list"] = data["demo_list"][:shot]
            input_data.append(data)
        
        self.data = input_data


def run(total=1,
        split=0,
        data_name="biggsm",
        distance_name="seuclidean",
        model_type="gpt",
        model_name="gpt-3.5-turbo",
        api_key="sk-xxx",
        request_proxy=True,
        max_tokens=1200,
        temperature=0.1,
        shot=5):
    test_data_path = f"experiments/theory-verification/icl-verification/data/biggsm_test_{distance_name}.jsonl"
    save_dir = f"experiments/theory-verification/icl-verification/request_data"
    if data_name == "biggsm":
        train_data = read_jsonl(f"data/gsm8k/train.jsonl")
    else:
        train_data = read_jsonl(f"data/{data_name}/train.jsonl")
    model_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    asyncio.run(request_LLM(
        total=total,
        split=split,
        dataset=DataLoader(test_data_path=test_data_path, train_data=train_data, shot=shot),
        save_path=fr"{save_dir}/{distance_name}-tp-08.jsonl",
        consumer_size=25,
        create_prompt_fn=partial(create_prompt, prompt_config=None),
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        enable_multi_turn = False,
        request_proxy=request_proxy,
        return_origin=True,
        model_config=model_config
        ))

if __name__ == "__main__":
    fire.Fire(run)
