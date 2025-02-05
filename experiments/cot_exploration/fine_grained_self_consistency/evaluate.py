import re
from utils.data_tool import GSM8KData
from utils.request_tool import RequestOutput
from utils.tools import evaluate_expression


def loop_judge(condition_list, input_str):
    for cond in condition_list:
        if cond in input_str:
            return True
    return False

def get_x_y(response_list, idx):
    origin_eqs = [s for s in re.findall(r'<<(.*)?>>', response_list.get_origin_input(idx)["answer"])]
    
    x = len(origin_eqs)
    operation_list = [operation for eq1 in origin_eqs for operation in re.findall(r'[\+\-\*/]', eq1.split("=")[0])]
    
    max_time = 0
    
    for eq0 in origin_eqs:
        value, max_dict = evaluate_expression(eq0.split("=")[0])
        if max_time < max_dict["time"]:
            max_time = max_dict["time"]
    
    x = max_time
    
    if len(operation_list) == len(origin_eqs):
        y = len([x for x in origin_eqs if not x.strip("0.").startswith("0")])
    else:
        y = len(operation_list)
    
    return x, y

def is_equal(pred, answer):
    return abs(abs(round(float(pred), 2)) - abs(round(answer, 2))) < 0.01


def get_map_answer(response_list, question):
    return response_list.get_pred_answer(response_list.search_by_question(question))

def judge_error(pred):
    try:
        float(pred)
    except:
        return False
    return True

def main():
    LOOP_INDEX = 1
    response_list = []

    # Randomly remove one of reasoning paths
    response_list.append(RequestOutput(fr"experiments/cot_exploration/fine_grained_self_consistency/result/loop_{LOOP_INDEX}/test-gpt3.5-tp-05.jsonl"))
    response_list.append(RequestOutput(fr"experiments/cot_exploration/fine_grained_self_consistency/result/loop_{LOOP_INDEX}/test-gpt3.5-tp-00.jsonl"))
    response_list.append(RequestOutput(fr"experiments/cot_exploration/fine_grained_self_consistency/result/loop_{LOOP_INDEX}/test-gpt3.5-tp-01.jsonl"))
    response_list.append(RequestOutput(fr"experiments/cot_exploration/fine_grained_self_consistency/result/loop_{LOOP_INDEX}/test-gpt3.5-tp-02.jsonl"))
    response_list.append(RequestOutput(fr"experiments/cot_exploration/fine_grained_self_consistency/result/loop_{LOOP_INDEX}/test-gpt3.5-tp-03.jsonl"))
    # response_list.append(RequestOutput(fr"experiments/cot_exploration/fine_grained_self_consistency/result/loop_{LOOP_INDEX}/test-gpt3.5-tp-08.jsonl"))
    total = 0
    correct = 0
    input_list = []
    for idx in range(len(response_list[0])):
        pred1 = response_list[0].get_pred_answer(idx)
        question = response_list[0].get_origin_input(idx)["question"]
        pred_list = [pred1] + [get_map_answer(r, question) for r in response_list[1:]] 
        max_dict = {}
        for p in pred_list:
            key = round(float(p), 2)
            if key not in max_dict:
                max_dict[key] = 0
            max_dict[key] += 1
        max_value, max_key = -1, -1
        for key in max_dict:
            if max_dict[key] > max_value:
                max_value = max_dict[key]
                max_key = key
        pred = max_key
        
        obj_ = GSM8KData(response_list[0].get_origin_input(idx))
        answer = obj_.get_answer()
        
        input_list.append(idx)
        total += 1
        if judge_error(pred) and abs(abs(round(float(pred), 2)) - abs(round(answer, 2))) < 0.01:
            correct += 1
        
    print(f"TOTAL ACC: {correct/total*100:.2f} TOTAL: {total}")
    
main()