import re

from utils.tools import read_jsonl
import textgrad as tg

def get_answer(input_str):
    pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', input_str.replace(",", "").strip(".").split("=")[-1])]
    if len(pred_list) == 0:
        pred1 = -1
    else:
        pred1 = pred_list[-1]
    return pred1

class DataLoader():
    def __init__(self, load_path) -> None:
        input_data = read_jsonl(load_path)
        self.input_data = []
        for data in input_data:
            question = tg.Variable(data["question"],
                        role_description="question to the LLM",
                        requires_grad=False)
            answer = tg.Variable(get_answer(data["answer"]), requires_grad=False, role_description="Correct answer")
            self.input_data.append((question, answer))

    
    def __getitem__(self, index):
        return self.input_data[index]