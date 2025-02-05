import re

from utils.request_tool import RequestOutput
from utils.tools import evaluate_expression




class PowerCalculator():

    def __init__(self, U, R_E, U_W) -> None:
        self.U, self.R_E, self.U_W = U, R_E, U_W
    
    def get_r(self, response_list, idx, enable_cot=True):
        input_data = response_list.data[idx]
        origin_eqs = [s for s in re.findall(r'<<(.*)?>>', input_data["origin"]["answer"])]
        
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
        M = 7.0
        SIGMA = 20000
        if enable_cot:
            return abs(r1/M + (r2 + SIGMA)/N - 0.4) * 2
        else:
            return abs(r1/M*4.5 + (r2 + SIGMA)/N*5.2 - 0.1) * 2

        
    def calculate_power(self,
                        response_list:RequestOutput=None, idx=0,
                        enable_cot=True,
                        return_dict=False,
                        input_obj=None):
        if input_obj is None:
            input_data = response_list.data[idx]
        else:
            input_data = input_obj
        R = self.get_r(response_list, idx, enable_cot=enable_cot)
        origin_data = input_data["origin"]
        if "demo_list" in origin_data:
            U_2 = self.U_W * sum([x["flux"] for x in origin_data["demo_list"]])
        else:
            U_2 = 0
            
        if U_2 > 0:
            P = (U_2 + self.U) * (U_2 + self.U) * self.R_E / ((self.R_E + R)*(self.R_E + R))
        else:
            P = (U_2 + self.U) * (U_2 + self.U) * self.R_E / ((self.R_E + R)*(self.R_E + R))
        if return_dict:
            return {"P": P, "U_ICL": U_2, "U_Model": self.U, "R_CoT": R, "R_0": self.R_E}
        return P
    
class RoBERTaPowerCalculator(PowerCalculator):
    def __init__(self) -> None:
        self.U = 95
        self.R_E = 2.5
        self.U_W = 100

class SkillPowerCalculator(PowerCalculator):
    def __init__(self) -> None:
        self.U = 60
        self.R_E = 2.5
        self.U_W = 100

U_DICT = {
    "gemini-1.5-pro": 85,
    "gemini-1.5-flash": 65,
    "glm-3-turbo": 50,
    "glm-4": 52,
    "o1-mini": 85,
    "gpt-3.5-turbo": 60,
    "gpt3.5": 60,
    "o1-preview": 95,
    "Qwen2.5-7B-Instruct": 60,
    "Qwen2.5-14B-Instruct": 60,
    "Qwen2.5-32B-Instruct": 65,
    "Qwen2.5-72B-Instruct": 70,
    "claude-3-5-sonnet-20240620": 80,
    "claude-3-haiku-20240307": 65,
    "claude-3-opus-20240229": 85,
    "claude-3-sonnet-20240229": 80,
}


U_W_DICT = {
    "bert": 0.15,
    "roberta": 0.5,
    "bge": 0.1,
    "skill": 100,
}
class AutoPowerCalculator(PowerCalculator):
    def __init__(self, reason_model, represent_model, distance_model="projection_length") -> None:
        self.U = U_DICT[reason_model]
        self.R_E = 2.5
        self.U_W = U_W_DICT[represent_model]
        if distance_model == "manhattan":
            self.U_W /= 200

class AutoMultiHopQAPowerCalculator(PowerCalculator):
    def __init__(self, reason_model, represent_model) -> None:
        self.U = U_DICT[reason_model]
        self.R_E = 2.5
        if represent_model == "roberta":
            self.U_W = 0.5
        else:
            raise ValueError(f"Unknown represent_model for {represent_model}")
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import pipeline
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.process_func = pipeline("ner", model=model, tokenizer=tokenizer)
    
    def get_r(self, response_list, idx, enable_cot=True):
        origin_eqs = response_list.get_origin_input(idx)["hop"]
        if len(origin_eqs) == 0:
            return -1, -1
        x = len(origin_eqs)
        knowledge_num = response_list.get_origin_input(idx).get("knowledge_num", None)
        if knowledge_num is None:
            ner_results = []
            for t in origin_eqs:
                knowledge_num = 0
                for entity in self.process_func(t):
                    if "B-" in entity["entity"]:
                        knowledge_num += 1
                ner_results.append(knowledge_num)
            y = max(ner_results)
            response_list.data[idx]["origin"]["knowledge_num"] = y
        else:
            y = knowledge_num
        r1 = y
        r2 = x
        N = 10
        M = 10
        SIGMA = 0.2
        if enable_cot:
            return abs(r1/M + (r2 + SIGMA)/N - 0.4)
        else:
            # return r1/M * (r2 + SIGMA)/N * 9.8 - 0.2
            return abs(r1/M*2 + (r2 + SIGMA)/N - 0.4)
        

class AutoMedProbPowerCalculator(PowerCalculator):
    def __init__(self, reason_model, represent_model) -> None:
        self.U = U_DICT[reason_model]
        self.R_E = 2.5
        if represent_model == "roberta":
            self.U_W = 0.5
        else:
            raise ValueError(f"Unknown represent_model for {represent_model}")
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import pipeline
        tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
        model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")
        self.process_func = pipeline("ner", model=model, tokenizer=tokenizer)
    
    def get_r(self, response_list, idx, enable_cot=True):
        if "step_num" in response_list.get_origin_input(idx):
            x = response_list.get_origin_input(idx)["step_num"]
            y = response_list.get_origin_input(idx)["entity_num"]
        else:
            origin_eqs = response_list.get_origin_input(idx)["exp"].split(". ")
            if len(origin_eqs) == 0:
                return -1
            x = len(origin_eqs)
            knowledge_num = response_list.get_origin_input(idx).get("knowledge_num", None)
            if knowledge_num is None:
                ner_results = []
                for t in origin_eqs:
                    knowledge_num = 0
                    for entity in self.process_func(t):
                        if "B-" in entity["entity"]:
                            knowledge_num += 1
                    ner_results.append(knowledge_num)
                y = max(ner_results)
                
                response_list.data[idx]["origin"]["knowledge_num"] = y
            else:
                y = knowledge_num
        if x < 0 or y < 0:
            return -1
        r1 = x
        r2 = y
        N = 10
        M = 10
        SIGMA = 0.2
        if enable_cot:
            return abs(r1/M + (r2 + SIGMA)/N - 0.1) * 0.4
        else:
            return abs(r1/M*2 + (r2 + SIGMA)/N - 0.1) * 0.4
        
class AutoRoboticPlanningPowerCalculator(PowerCalculator):
    def __init__(self, reason_model, represent_model) -> None:
        self.U = U_DICT[reason_model]
        self.R_E = 2.5
        if represent_model == "roberta":
            self.U_W = 1.5
        else:
            raise ValueError(f"Unknown represent_model for {represent_model}")
    
    def get_r(self, response_list, idx, enable_cot=True):
        plan_list = []
        if isinstance(response_list.data[idx]["ground_truth_plan"], str):
            plan_list = response_list.data[idx]["ground_truth_plan"].split("\n")
        else:
            plan_list = response_list.data[idx]["ground_truth_plan"]
        r1 = len(plan_list)
        r2 = max([len(x.split(" "))-1 for x in plan_list])
        N = 7.0
        M = 7.0
        SIGMA = 0.2
        if enable_cot:
            return abs(r1/M + (r2 + SIGMA)/N - 0.4) * 2
        else:
            return abs(r1/M*2 + (r2 + SIGMA)/N - 0.4) * 2