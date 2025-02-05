
import re

from utils.data_tool import GSM8KData

def judge_error(pred):
    try:
        float(pred)
    except:
        return False
    return True

class CorrectCalculator():
    def calculate_correct(self, response_list, idx):
        pred = response_list.get_pred_answer(idx)
        obj_ = GSM8KData(response_list.data[idx]["origin"])
        answer = obj_.get_answer()
        C = 0
        if judge_error(pred) and abs(abs(round(float(pred), 2)) - abs(round(answer, 2))) < 0.01:
            C = 1
        return C

    def calculate_correct_by_str(self, response, origin):
        pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', response.replace(",", "").strip(".").split("=")[-1])]
        if len(pred_list) == 0:
            pred = -1
        else:
            pred = pred_list[-1]
        obj_ = GSM8KData(origin)
        answer = obj_.get_answer()
        C = 0
        if judge_error(pred) and abs(abs(round(float(pred), 2)) - abs(round(answer, 2))) < 0.01:
            C = 1
        return C
    
class MultiHopQACorrectCalculator():
    def calculate_correct(self, response_list, idx):
        pred1 = response_list.get_text_answer(idx)
        obj_ = GSM8KData(response_list.data[idx]["origin"])
        answer = obj_.get_text_answer()
        C = 0
        if pred1.lower().strip() == answer.lower().strip():
            C = 1
        return C
    
class RoboticPlanningCorrectCalculator():
    def calculate_correct(self, response_list, idx):
        origin_data = response_list.data[idx]
        C = 1 if origin_data["llm_correct"] else 0
        return C
ALPHA_LIST = ["NONE", "A", "B", "C", "D"]
class MedProbCorrectCalculator():
    def calculate_correct(self, response_list, idx):
        pred = response_list.get_text_answer(idx)
        if "[[Answer]]" in pred:
            pred = pred.split("[[Answer]]")[-1]
        find_list = re.findall(r"\([A-D]\)", pred)
        if len(find_list) > 0:
            pred = find_list[-1].strip("(").strip(")")
        else:
            pred = "NONE"
        answer = ALPHA_LIST[response_list.get_origin_input(idx)["cop"]]
        C = 0
        if pred.lower().strip() == answer.lower().strip():
            C = 1
        return C