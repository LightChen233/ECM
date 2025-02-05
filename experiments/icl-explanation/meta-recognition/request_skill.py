import asyncio
from functools import partial
import os
import random
import fire

from utils.request_tool import request_LLM
from utils.tools import read_jsonl

random.seed(42)
#PAWS-X Prompt

# DATA_PATH = "data/biggsm/data_cleaned.jsonl"
DATA_INDEX = "biggsm"
DATA_SPLIT = "test"
MODEL_INDEX = "gpt4"

DATA_PATH = f"data/{DATA_INDEX}/{DATA_SPLIT}.jsonl"
MODEL_DICT = {
    "gpt35": {
        "model_name": "gpt-3.5-turbo",
        "api_key": "sk-xxx",
    },
    "gpt4": {
        "model_name": "gpt-4o",
        "api_key": "sk-xxx",
    }
}



def create_prompt(data, prompt_config):
    instruction = f"""Consider this mathematical question. Label this question with a mathematical skill that would be required to solve the question. Basically, you should be able to use the skill as a dictionary key in python. The skill name should be lower case letters only. The skill name should be very descriptive and you may use multiple words to describe the skills required in the question. If you do use multiple words per question, then join them by an underscore.

[QUESTION]
{data['question']}

Your answer should be as follows: [REASON] <reason for the skill>, [SKILL] <name of the skill>"""
# Question: A Statistics student wants to find out the average daily allowance of the middle school students. According to his survey, 2/3 of the students receive an average of $6 allowance per day while the rest gets an average of $4 a day. If he surveyed 60 students, what is the total amount of money those 60 students get in a day?
# Answer: There are 60 students x 2/3 = <<60*2/3=40>>40 students who have a $6 daily allowance.\nWhile there are 60 students - 40 students = <<60-40=20>>20 students who have a $4 daily allowance.\nThe sum of the allowances of the 40 students who received $6 daily is 40 students x $6/day = $<<40*6=240>>240.\nThe sum of the allowances of the 20 students who received $4 daily is 20 students x $4/day = $<<20*4=80>>80.\nThe total daily amount of money of those 60 students is $240 + $80 = $<<240+80=320>>320.\n#### 320"""
# There are 25 roses in a garden. There are 40 tulips. There are 35 daisies. What percentage of flowers are not roses?", "answer": "There are 25+40+35=<<25+40+35=100>>100 flowers total.\nThere are 40+35=<<40+35=75>>75 flowers that are not roses.\nTherefore, (75/100)*100=<<(75/100)*100=75>>75% of the flowers are not roses.\n#### 75"}
    # instruction += data["question"]  + f"\n\nLet's think step-by-step!"
    # instruction += str(data["x"]) + " " + data["operation"] + " " + str(data["y"]) + "="
    return instruction


class MyData2:
    def __init__(self, load_path=DATA_PATH) -> None:
        input_data = []
        for i, data in enumerate(read_jsonl(load_path)):
            if "index" not in data:
                data["index"] = str(i)
            input_data.append(data)
            
        self.data = input_data


def run(total=1, # 用于启动多进程脚本
        split=0, #
        model_type="gpt",
        model_name="gpt-4-turbo",
        api_key="sk-xxx",
        request_proxy=True,
        max_tokens=600,
        operation="+",
        size=50,
        temperature=0.1):
    model_name = MODEL_DICT[MODEL_INDEX]["model_name"]
    api_key = MODEL_DICT[MODEL_INDEX]["api_key"]
    model_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    operation_map = {
        "*": "time",
        "+": "plus",
        "-": "minus",
        "/": "divide",
        
    }
    if not os.path.exists(fr"experiments/ICL/skill/{DATA_INDEX}"):
        os.makedirs(fr"experiments/ICL/skill/{DATA_INDEX}",exist_ok=True)
    asyncio.run(request_LLM(
        total=total,
        split=split,
        dataset=MyData2(DATA_PATH),
        save_path=fr"experiments/ICL/skill/{DATA_INDEX}/{DATA_SPLIT}-{MODEL_INDEX}-tp-01-1.jsonl",
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
