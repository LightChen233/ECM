from copy import deepcopy
import json
import random
import re

from tqdm import tqdm
from utils.tools import read_jsonl, write_jsonl
random.seed(42)


# Part of the code is modified from the code snippets provided in "Solving Quantitative Reasoning Problems with Language Models" by Lewkowycz et al.
import re
import sympy
from sympy.parsing.latex import parse_latex

SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), ('\\(', ''), ('\\)', ''), ('\\[', ''), ('\\]', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}'),
    ('\\right)', ')'), ('\\le(', '(')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\\mathrm{th}',
    r'^\\circ', r'^\circ', r'^{\\circ}', r'\;', r',\!', '{,}', '"', '\\dots', "\\end{aligned}", "\\end{aligned", "\\end{align*}", "\\end{align*", "x\\in", "x=", "x =", "x \\in", "%", "%"
]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    print_str = final_answer
    
    
    if "\\boxed{" in final_answer:
        my_stack = []
        my_str = ""
        for s in final_answer.strip().strip(".").strip().split("\\boxed")[-1]:
            my_str += s
            if s == "{":
                my_stack.append("{")
            elif s== "}":
                my_stack.pop()
                if len(my_stack) == 0:
                    break
        
        if len(my_stack) == 0:
            final_answer = my_str[1:-1]
    else:
        final_answer = final_answer.split('=')[-1].strip().strip(".").strip()
        if "\\[" in final_answer:
            answer_list = re.findall(r'\\\[(.*)\\\]', final_answer.replace("\n",""))
            if len(answer_list) > 0:
                final_answer = answer_list[-1]
        elif "\\(" in final_answer:
            answer_list = re.findall(r'\\\((.*)\\\)', final_answer.replace("\n",""))
            if len(answer_list) > 0:
                final_answer = answer_list[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\\()(.*)(\\\))', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer.strip(".").strip()

def check_sympy_equivalence(formatted_target_str, formatted_prediction_str):
    if formatted_target_str.strip().replace("{", "").replace("}", "") == formatted_prediction_str.strip().replace("{", "").replace("}", ""):
        return True
    flag = False    
    try:
        target_expr = parse_latex(formatted_target_str)
    except:
        target_expr = formatted_target_str
        flag = True
    
    try:
        prediction_expr = parse_latex(formatted_prediction_str)
    except:
        prediction_expr = formatted_prediction_str
        flag = True
    
    if flag == True:
        return formatted_target_str == formatted_prediction_str

    try:
        return sympy.simplify(target_expr - prediction_expr) == 0
    except:
        return False

class GSM8KData():
    def __init__(self, obj) -> None:
        self.obj = obj
        self.obj, self.equation_list = self.extract_equation(self.obj)
    
    def get_answer(self):
        res_str = self.obj["answer"].replace(",", "").strip(".").split("\n#### ")[-1]
        try:
            return round(float(res_str), 2)
        except:
            return -1
    def get_text_answer(self):
        return self.obj["answer"]
    def extract_equation(self, obj):
        exp_list = [s for s in re.findall(r'<<(.*)?>>', obj["answer"])]
        equation_list = []
        # num_list = [s for s in exp_list ]
        obj["operation"] = {"+": 0, "-": 0, "*": 0, "/": 0}
        for exp in exp_list:
            exp = exp.strip(".0").strip(".00")
            # exp = exp.replace(".0", "")
            # if "." in exp.split("=")[-1].strip().strip(".") or "/" in exp.split("=")[-1].strip():
            #     return obj, None
            ans = exp.split("=")[-1].strip()
            exp = exp.split("=")[0]
            operations = re.findall(r"\+|\-|\*|\/", exp)
            for operation in operations:
                obj["operation"][operation] += 1
            if ans == "":
                ans = "0"
            equation_list.append({"func": exp, "ans": ans})
        return obj, equation_list 

    # def parse_slot(self, input_str, num_list):
    #     answer_list = [input_str]
    #     for slot_idx, num in enumerate(num_list):
    #         temp_answer_list = []
    #         # ["..x.."]
    #         for ta in answer_list:
    #             # ["..", ".."] ->["..", "x", ".."]
    #             if "[[" not in ta and "." not in str(num):
    #                 al = ta.split(str(num))
    #                 for jdx, a in enumerate(al):
    #                     temp_answer_list.append(a)
    #                     if jdx != len(al) - 1:
    #                         temp_answer_list.append("[[SLOT_" + str(slot_idx) + "]]")
    #             else:
    #                 temp_answer_list.append(ta)
    #         answer_list = temp_answer_list
    #     return "".join(temp_answer_list)
    def parse_slot(self, input_str, num_list):
        answer_list = re.split(r'((?<![0-9.])-?\d+\.?\d*)', input_str)
        for slot_idx, num in enumerate(num_list):
            # ["..x.."]
            for token_idx, ta in enumerate(answer_list):
                # ["..", ".."] ->["..", "x", ".."]
                if "[[" not in ta and "." not in str(num):
                    if ta.endswith(".") and ta.replace(".", "") == str(num):
                        answer_list[token_idx] = "[[SLOT_" + str(slot_idx) + "]]"
                    elif ta == str(num):
                        answer_list[token_idx] = "[[SLOT_" + str(slot_idx) + "]]"
                    
                
            
        return "".join(answer_list)
    
    def get_abstract_data(self):
        temp_res = deepcopy(self.obj)
        num_list = []
        for s in re.findall(r'(?<![0-9.])-?\d+\.?\d*', temp_res["answer"].replace(",", "").strip(".")):
            if s not in num_list:
                num_list.append(s)
        # if len(num_list) != len(set(num_list)):
        #     return None
        num_map = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
        }
        for num in num_map:
            if num in temp_res["question"].replace(",", " ").split(" ") and num_map[num] in num_list:
                num_list.remove(num_map[num])
        temp_res["answer"] = self.parse_slot(temp_res["answer"], num_list)
        temp_res["question"] = self.parse_slot(temp_res["question"], num_list)
        
        temp_res["slot_num"] = len(num_list)
        return GSM8KData(temp_res)
    
    def fulfill_abstract(self):
        if self.equation_list is None:
            return None
        res = deepcopy(self.obj)
        slot_dict = {}
        if "slot_num" in self.obj and self.obj["slot_num"] > 0:
            for eq in self.equation_list:
                func_str = eq["func"]
                func_var = re.split('([+\-*/()])', func_str)
                # if "." in func_var:
                #     return None
                for jdx, slot in enumerate(func_var):
                    if slot in "+-/*()":
                        continue
                    if "[[" not in slot:
                        continue
                    if slot in slot_dict:
                        value = slot_dict[slot]
                    else:
                        if jdx != 0:
                            operation = func_var[jdx-1]
                            if operation in "()" or operation == "":
                                operation = "*"
                        else:
                            operation = "*"
                        if operation == "*":
                            value = random.choice([random.randint(1, 2000), random.randint(1, 200), random.randint(1, 20)])
                            # value = random.randint(0, 2000)
                        elif operation == "/":
                            value = random.randint(1, 2000)
                        elif operation == "-":
                            value = random.randint(0, 2000)
                        elif operation == "+":
                            value = random.randint(0, 2000)
                        slot_dict[slot] = value
                    func_str = func_str.replace(slot, str(value))
                try:
                    answer = eval(func_str)
                    if int(answer) != answer:
                        answer = round(answer, 2)
                except:
                    return None
                slot_dict[eq["ans"]] = answer
            for slot in slot_dict:
                res["answer"] = res["answer"].replace(slot, str(slot_dict[slot]))
                res["question"] = res["question"].replace(slot, str(slot_dict[slot]))
            if "[[" in res["answer"] or "[[" in res["question"]:
                return None
        return GSM8KData(res)
    
    def __str__(self) -> str:
        return json.dumps(self.obj)


class GSM8KDataList():
    def __init__(self) -> None:
        self.data_list = [GSM8KData(x) for x in read_jsonl("gsm8k/test.jsonl")]

if __name__ == "__main__":
    data_list = [GSM8KData(x) for x in read_jsonl("data/gsm8k/test.jsonl")]
    res_list = []
    for tdx, data in enumerate(tqdm(data_list)):
        if tdx == 4:
            print(1)
        data = data.get_abstract_data()
        if data is None or data.equation_list is None:
            continue
        idx = 0
        max_try = 0
        while idx < 5:
            try:
                if tdx == 4 and idx == 0:
                    print(1)
                temp_data = data.fulfill_abstract()
                if max_try > 10:
                    idx+= 1
                    max_try = 0
                    continue
                if temp_data is None or temp_data.equation_list is None:
                    max_try += 1
                    continue
                if len([eq for eq in temp_data.equation_list if "/" in eq["func"]]) > 0:
                    max_try += 1
                    continue
                plus_num = max([abs(float(eq["ans"])) for eq in temp_data.equation_list if "+" in eq["func"] or "-" in eq["func"]] + [0])
                times_num = max([abs(float(eq["ans"])) for eq in temp_data.equation_list if "*" in eq["func"]] + [1])
                if plus_num == 0 and times_num == 1:
                    max_try += 1
                    continue
                
                if times_num > 20000:
                    print("Error Time")
                    max_try += 1
                    continue
                if plus_num <15 or plus_num > 1e5:
                    print("Error PLUS")
                    max_try += 1
                    continue
            
                idx+= 1
                print(temp_data.obj)
                res_list.append(temp_data.obj)
            except:
                max_try += 1
                if max_try > 10:
                    idx+= 1
                    max_try = 0
                    continue
    write_jsonl("data/gsm8k/large_test.jsonl", res_list, "a")