import re
from matplotlib import pyplot as plt
from pandas import DataFrame
from utils.data_tool import GSM8KData
from utils.request_tool import RequestOutput
from utils.tools import evaluate_expression
from scipy import stats
import seaborn as sns

def loop_judge(condition_list, input_str):
    for cond in condition_list:
        if cond in input_str:
            return True
    return False

def get_r(response_list, idx, mode="natural language"):
    origin_eqs = [s for s in re.findall(r'<<(.*)?>>', response_list.get_origin_input(idx)["answer"])]
    
    x = len(origin_eqs)
    operation_list = [operation for eq1 in origin_eqs for operation in re.findall(r'[\+\-\*/]', eq1.split("=")[0])]
    
    max_time = 0
    sum_time = 0
    for eq0 in origin_eqs:
        value, max_dict = evaluate_expression(eq0.split("=")[0])
        if max_time < max_dict["time"]:
            max_time = max_dict["time"]
        sum_time += max_dict["time"]
    
    
    if len(operation_list) == len(origin_eqs):
        r1 = len([x for x in origin_eqs if not x.strip("0.").startswith("0")])
    else:
        r1 = len(operation_list)
    r2 = max_time
    N = 1.6e5
    K = 0.106
    M = 7.0
    SIGMA = 20000
    if mode == "tool":
        return abs(r1/M - 0.2)
    elif mode == "pot":
        # print(abs(r1/M + 0.5)/1.8)
        # return abs(r1/M - 0.1)/1.3
        return abs(r1/M - 0.2)/1.2
    else:
        return abs(r1/M + (r2 + SIGMA)/N - 0.4)
    
def is_equal(pred, answer):
    return abs(abs(round(float(pred), 2)) - abs(round(answer, 2))) < 0.01


def judge_error(pred):
    try:
        float(pred)
    except:
        return False
    return True

def draw_points(data_path, U, R_E, I_W, parse_mode, mode, split_name):
    response_list = RequestOutput(data_path, auto_index=False)
    if parse_mode == "tool" and mode == "tool":
        STEP = 0.8
    elif parse_mode == "pot" and mode == "tool":
        STEP = 0.2
    elif parse_mode == "pot":
        STEP = 0.5
    else:
        STEP = 40
    res_list = []
    for idx in range(len(response_list)):
        # pred1 = response_list.get_parsed_pred_answer(idx)
        if parse_mode == "tool":
            pred1 = response_list.get_parsed_pred_answer(idx)
        elif parse_mode == "pot":
            pred1 = response_list.get_program_answer(idx)
        else:
            pred1 = response_list.get_pred_answer(idx)
        R = get_r(response_list, idx, mode=mode)
        
        R = R*2
        origin_data = response_list.get_origin_input(idx)
        if "demo_list" in origin_data:
            U_2 = I_W * sum([x["flux"] for x in origin_data["demo_list"]]) #/len(response_list.get_origin_input(idx)["demo_list"])
        else:
            U_2 = 0
        # print(U_2)
        P = (U_2 + U) * (U_2 + U) * R_E / ((R_E + R)*(R_E + R))
        print(P)
        obj_ = GSM8KData(origin_data)
        answer = obj_.get_answer()
        C = 0
        if judge_error(pred1) and abs(abs(round(float(pred1), 2)) - abs(round(answer, 2))) < 0.01:
            C = 1
        res_list.append({"R": R, "U_2": U_2, "U": U, "R_E": R_E, "P": P, "C": C, "index": origin_data["index"]})
    correct_list = {}
    idx_list = {}
    p_list = {res["index"]: res['P'] for res in res_list}
    print(f"{split_name}\tAVG P: {sum([p_list[key] for key in p_list])/len(p_list)}")
    for res in res_list:
        key = int(res["P"]/STEP)
        # print(key)
        if key not in correct_list:
            correct_list[key] = []
            idx_list[key] = []
        correct_list[key].append(res["C"])
        # if res["C"] == 1:
        idx_list[key].append(res["index"])
    key_list = list(correct_list.keys())
    draw_data = {"acc": [], "p": [], "index": []}
    total = 0
    correct = 0
    for key in sorted(key_list):
        total += len(correct_list[key])
        correct += sum(correct_list[key])
        if len(correct_list[key]) > 7:
            draw_data["p"].append(key*STEP)
            
            draw_data["acc"].append(sum(correct_list[key])/len(correct_list[key]))
            draw_data["index"].append(idx_list[key])
            # print(f"{key*STEP}    {sum(correct_list[key])/len(correct_list[key])}\t TOTAL: {len(correct_list[key])}")
    print("Total: ", total, " Correct: ", correct, " Acc: ", round(correct/total*100, 2))
    return draw_data, p_list

def main():
    sns.set_theme(style="ticks")
    U = 60
    R_E = 2.5
    I_W = 0.5
    
    PATH_DICT = {
        "tool-cot": {
            "data_path": "experiments/cot-explanation/tool-usage/request_data/tool.jsonl",
            "parse_mode": "tool",
            "mode": "tool",
        },
        "tool-cot-nl": {
            "data_path": "experiments/cot-explanation/tool-usage/request_data/tool.jsonl",
            "parse_mode": "tool",
            "mode": "natural language",
        },
        "pot": {
            "data_path": "experiments/cot-explanation/tool-usage/request_data/pot.jsonl",
            "parse_mode": "pot",
            "mode": "pot",
        },
        "pot-tool": {
            "data_path": "experiments/cot-explanation/tool-usage/request_data/pot.jsonl",
            "parse_mode": "pot",
            "mode": "tool",
        },
    }
    res_data = {"acc": [], "p": [], "class": []}
    
    dict_p_list = {}
    for key in PATH_DICT:
        
        draw_data, p_list = draw_points(PATH_DICT[key]["data_path"], U, R_E, I_W, PATH_DICT[key]["parse_mode"], PATH_DICT[key]["mode"], split_name=key)
        dict_p_list[key] = p_list
        for i, (acc, p, idx) in enumerate(zip(draw_data["acc"], draw_data["p"], draw_data["index"])):
            res_data["acc"].append(acc)
            res_data["p"].append(p)
            res_data["class"].append(key)
    print(stats.spearmanr(res_data["acc"], res_data["p"]))
    color_list = [(98,40,158), (200,106,114), (239,175,83)]
    color_list.reverse()
    color_list = [(x/256.0, y/256.0, z/256.0) for x, y, z in color_list]
    for key in PATH_DICT:
        print(f"{key}:\t", stats.spearmanr([x for _i, x in enumerate(res_data["acc"]) if res_data["class"][_i] == key], [x for _i, x in enumerate(res_data["p"]) if res_data["class"][_i] == key]))
    g = sns.lmplot(
        data=DataFrame(res_data),
        x="acc", y="p", hue="class",
        height=5, robust=True, 
    )
    plt.savefig('experiments/cot-explanation/tool-usage/result.svg', format='svg')
    plt.show()
main()