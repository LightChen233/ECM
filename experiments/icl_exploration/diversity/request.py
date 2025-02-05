import asyncio
from functools import partial
import os
import random
import fire

from utils.request_tool import request_LLM
from utils.tools import read_jsonl

random.seed(42)

DATA_INDEX = "biggsm"
DATA_SPLIT = "test"
MODEL_INDEX = "gpt35"

DATA_PATH = f"semantic/roberta_embed/biggsm_test.jsonl"
train_data = read_jsonl(f"data/biggsm/train.jsonl")

SHOT = 5


def create_prompt(data, prompt_config):
    
    instruction = f""""""
    for d in data["request_list"]:
        if len(d['answer']) > 1500:
            d['answer'] = d['answer'][:1500]
        instruction += f"""[QUESTION]
{d['question']}
[ANSWER]
{d['answer']}

"""
    instruction += f"""[QUESTION]
{data['question']}"""
    return instruction


class MyData2:
    def __init__(self, load_path=DATA_PATH, top_x=5) -> None:
        input_data = []
        for i, data in enumerate(read_jsonl(load_path)):
            if "index" not in data:
                data["index"] = str(i)
            del data["embed"]
            data["request_list"] = []
            temp_demo_list = []
            for j, d in enumerate(data["demo_list"][:top_x]):
                if j % (int(top_x / SHOT)) == 0:
                    temp_demo_list.append(d)
            if len(temp_demo_list) < SHOT:
                start_idx = -1
                while(len(temp_demo_list) < SHOT):
                    temp_demo_list.append(temp_demo_list[start_idx])
                    start_idx -= 1
                # print(len(temp_demo_list))
            if len(temp_demo_list) > SHOT:
                temp_demo_list = temp_demo_list[:SHOT]
            for d in temp_demo_list:
                data["request_list"].append(train_data[int(d["index"])])
            data["demo_list"] = temp_demo_list
            input_data.append(data)
        
        self.data = input_data


def run(total=1, # 用于启动多进程脚本
        split=0, #
        model_type="gpt",
        model_name="gpt-3.5-turbo",
        api_key="sk-xxx",
        request_proxy=True,
        max_tokens=2000,
        temperature=0.1,
        top=2000):
    TOP_X = top
    model_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    temperature_str = str(round(temperature,1)).replace(".", "")
    if not os.path.exists(fr"experiments/icl_exploration/diversity/request_data/dd-icl/res-{temperature_str}-1"):
        os.makedirs(fr"experiments/icl_exploration/diversity/request_data/dd-icl/res-{temperature_str}-1",exist_ok=True)
    
    asyncio.run(request_LLM(
        total=total,
        split=split,
        dataset=MyData2(DATA_PATH, TOP_X),
        save_path=fr"experiments/icl_exploration/diversity/request_data/dd-icl/res-{temperature_str}-1/diverse-gpt35-temp-{temperature_str}-top-{TOP_X}.jsonl",
        consumer_size=60,
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
