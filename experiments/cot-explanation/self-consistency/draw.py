import os
import re
from matplotlib import pyplot as plt
from pandas import DataFrame
from tqdm import tqdm
from utils.data_tool import GSM8KData
from utils.request_tool import RequestOutput
from utils.tools import evaluate_expression, read_jsonl, write_jsonl
import seaborn as sns

def get_r(response_list, idx, enable_cot=True):
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
    if enable_cot:
        return abs(r1/M + (r2 + SIGMA)/N - 0.4)
    else:
        # return r1/M * (r2 + SIGMA)/N * 9.8 - 0.2
        return abs(r1/M*2 + (r2 + SIGMA)/N - 0.4)
    
def is_equal(pred, answer):
    return abs(abs(round(float(pred), 2)) - abs(round(answer, 2))) < 0.01


def get_map_answer(response_list, question):
    return response_list.get_program_answer(response_list.search_by_question(question))

def judge_error(pred):
    try:
        float(pred)
    except:
        return False
    return True

def draw_points(response_list, U, R_E, I_W, enable_cot, consistency_size):
    res_list = []
    for idx in range(len(response_list)):
        oringin_R = get_r(response_list, idx, enable_cot=enable_cot)
        # print(R)
        R = oringin_R*2/consistency_size + 2.5
        target_R = oringin_R*2/consistency_size
        origin_data = response_list.get_origin_input(idx)
        if "demo_list" in origin_data:
            U_2 = I_W * sum([x["flux"] for x in origin_data["demo_list"]])
        else:
            U_2 = 0
            
        
        P = (U_2 + U) * (U_2 + U) * R_E / ((R_E + R)*(R_E + R))
        target_P = (U_2 + U) * (U_2 + U) * R_E / ((R_E + target_R)*(R_E + target_R))
        
        # print(P)
        if P > 70000:
            continue
        obj_ = GSM8KData(origin_data)
        answer = obj_.get_answer()
        prediction_dict = {}
        for content in response_list.data[idx]["pred"][1]['content'][:consistency_size]:
            pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', content["text"].replace(",", "").strip(".").split("=")[-1])]
            if len(pred_list) == 0:
                pred1 = -1
            else:
                pred1 = pred_list[-1]
            pred_str = str(round(float(pred1), 2))
            if pred_str not in prediction_dict:
                prediction_dict[pred_str] = 0
            prediction_dict[pred_str] += 1
        max_key = "-1"
        max_value = -1
        pass_value = 0
        for pred_key in prediction_dict:
            if prediction_dict[pred_key] > max_value:
                max_value = prediction_dict[pred_key]
                max_key = pred_key
            if judge_error(pred1) and abs(abs(round(float(pred_key), 2)) - abs(round(answer, 2))) < 0.01:
                pass_value = 1
        C = 0
        
        if judge_error(pred1) and abs(abs(round(float(max_key), 2)) - abs(round(answer, 2))) < 0.01:
            C = 1
        res_list.append({"R": R, "U_2": U_2, "U": U, "R_E": R_E, "P": P, "C": C, "pass": pass_value, "pass_p": target_P, "index": origin_data["index"]})
    correct_list = []
    idx_list = []
    p_list = []
    pass_list = []
    target_p_list = []
    for res in res_list:
        p_list.append(res["P"])
        correct_list.append(res["C"])
        pass_list.append(res["pass"])
        target_p_list.append(res["pass_p"])
        if res["C"] == 1:
            idx_list.append(res["index"])
    # response_list.save("/Users/chenqiguang/Desktop/code/electric-exp/experiments/ICL/self-consistency/unified-1.jsonl")
    draw_data = {"acc": [], "p": [], "pass": [], "pass@k": [], "pass_p": [], "index": []}
    draw_data["p"].append(sum(p_list)/len(p_list))
    draw_data["acc"].append(sum(correct_list)/len(correct_list))
    draw_data["pass@k"].append(sum(pass_list)/len(pass_list))
    draw_data["pass_p"].append(sum(target_p_list)/len(target_p_list))
    draw_data["index"].append(idx_list)
            # print(f"{key*STEP}    {sum(correct_list[key])/len(correct_list[key])}\t TOTAL: {len(correct_list[key])}")
    return draw_data

def dict2list(dict_obj):
    key_list = list(dict_obj.keys())
    size = len(dict_obj[key_list[0]])
    res_data = []
    for i in range(size):
        temp_obj = {}
        for key in dict_obj:
            temp_obj[key] = dict_obj[key][i]
        res_data.append(temp_obj)
    return res_data

def list2dict(list_obj):
    key_list = list(list_obj[0].keys())
    res_data = {key: [] for key in key_list}
    for d in list_obj:
        for key in key_list:
            res_data[key].append(d[key])
    return res_data

def main():
    sns.set_theme()
    U = 60
    R_E = 2.5
    I_W = 100
    # response_list = RequestOutput(fr"/Volumes/My Passport/code/electric-exp/experiments/electric/semantic-result/biggsm/test-gpt35-5-tp-00.jsonl")
    # response_list = RequestOutput(fr"experiments/electric/skill-result/biggsm/test-gpt35-5-tp-18-reverse.jsonl", auto_index=False)
    ### ICL+COT
    PATH_DICT = {
        "tp05": {
            "data_path": "experiments/cot-explanation/self-consistency/request_data/gpt35-self-consist-tp-05.jsonl",
            "enable_cot": True,
        },
        "tp08": {
            "data_path": "experiments/cot-explanation/self-consistency/request_data/gpt35-self-consist-tp-08.jsonl",
            "enable_cot": True,
        },
        "tp01": {
            "data_path": "experiments/cot-explanation/self-consistency/request_data/gpt35-self-consist-tp-01.jsonl",
            "enable_cot": True,
        },
    }
    save_path = "experiments/cot-explanation/self-consistency/request_data/res_data.jsonl"
    if os.path.exists(save_path):
        res_data = list2dict(read_jsonl(save_path))
    else:
        res_data = {"value": [], "class": [], "size": []}
        
        
        for key in PATH_DICT:
            response_list = RequestOutput(PATH_DICT[key]["data_path"], auto_index=False)
            for size in tqdm(range(1, len(response_list.data[0]["pred"][1]['content'])+1)): # 
                draw_data = draw_points(response_list, U, R_E, I_W, PATH_DICT[key]["enable_cot"], consistency_size=size)
                for i, (acc, p, pass_acc, pass_p, idx) in enumerate(zip(draw_data["acc"], draw_data["p"], draw_data["pass@k"], draw_data["pass_p"], draw_data["index"])):
                    # print(acc)
                    # print(norm_p(p))
                    res_data["value"].append(acc)
                    # res_data["p"].append(p)
                    res_data["size"].append(size)
                    res_data["class"].append("acc")
                    res_data["value"].append(p)
                    res_data["size"].append(size)
                    res_data["class"].append("p")
                    res_data["value"].append(pass_acc)
                    res_data["size"].append(size)
                    res_data["class"].append("pass_acc")
                    res_data["value"].append(pass_p)
                    res_data["size"].append(size)
                    res_data["class"].append("pass_p")
    if not os.path.exists(save_path):
        write_jsonl(save_path, dict2list(res_data), "w")
    fig, ax1 = plt.subplots()
    ax1.set_ylim(0.3, 0.9)
    # Split p
    acc_res_data = list2dict([x for x in dict2list(res_data) if x["class"] in ["acc", "pass_acc"]])
    p_res_data = list2dict([x for x in dict2list(res_data) if x["class"] in ["p", "pass_p"]])
    sns.lineplot(x="size", y="value",
             hue="class", 
             ax=ax1,
             data=DataFrame(acc_res_data))
    ax2 = ax1.twinx()
    # ax2.set_ylim(3000, 6800)
    ax2.set_ylim(-2200, 5000)
    sns.lineplot(x="size", y="value",
             hue="class", 
             ax=ax2,
             data=DataFrame(p_res_data))
    plt.savefig('experiments/cot-explanation/self-consistency/result.svg', format='svg')
    plt.show()
    
main()