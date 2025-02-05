import argparse
import asyncio
from functools import partial
import json
import os
import random
import re
import fire

from utils.request_tool import request_LLM
from utils.tools import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--step", default=1, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
args = parser.parse_args()

random.seed(42)
def create_prompt(data, prompt_config):
    instruction = """"""
    for d in data["demo"]:
        instruction += f"""[[Question]]
{d['question']}
[[Answer]]
{d['answer']}

"""
    
    instruction += "[[Question]]\n" + data["question"]
    return instruction

STEP = args.step
context_data = read_jsonl("experiments/icl_exploration/demo_optimize/request_data/origin_prompt.jsonl")
last_pred_text = context_data[STEP]
last_pred_text = re.sub(r'(\\text)\{(.*?)\}', r'\1\\{\2\\}', last_pred_text)
pred_list = [s for s in re.findall(r'((?<!\\)\{.*?(?<!\\)\})', last_pred_text.replace(",\n    },\n    {\n        ", "\n    },\n    {\n        ").replace(",\n    }", "\n    }").replace("\n    },\n   ]", "\n    }\n   ]"), re.M|re.I|re.S)]
try:
    pred1 = [json.loads(p) for p in pred_list]
except:
    print(-1)
class MyData2:
    def __init__(self, load_path="") -> None:
        input_data = []
        
        for i, data in enumerate(read_jsonl(load_path)):
            if "index" not in data:
                data["index"] = str(i)
            data["demo"] = pred1
            input_data.append(data)
            
        self.data = input_data


def run(total=1, # 用于启动多进程脚本
        split=0, #
        model_type="gpt",
        model_name="gpt-3.5-turbo",
        api_key="sk-xxx",
        request_proxy=True,
        max_tokens=1000,
        temperature=args.temperature):
    
    model_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    temp_str = str(temperature).replace(".", "")
    print(temperature, temp_str)
    if not os.path.exists(fr"experiments/demo_optimize/query_optimize/result/origin-{temp_str}"):
        os.makedirs(fr"experiments/demo_optimize/query_optimize/result/origin-{temp_str}", exist_ok=True)
    asyncio.run(request_LLM(
        total=total,
        split=split,
        dataset=MyData2("data/biggsm/test.jsonl"),
        save_path=fr"experiments/demo_optimize/query_optimize/result/origin-{temp_str}/unified-{STEP}-tp-{temp_str}.jsonl",
        consumer_size=60,
        create_prompt_fn=partial(create_prompt, prompt_config=None),
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        enable_multi_turn = True,
        request_proxy=request_proxy,
        return_origin=True,
        model_config=model_config
        ))

if __name__ == "__main__":
    fire.Fire(run)
