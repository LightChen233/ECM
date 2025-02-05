import json
import os


def read_jsonl(data_path):
    input_data = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                input_data.append(json.loads(line.strip()))
    else:
        print(f"Missing {data_path}")
    return input_data


def write_jsonl(save_path, save_object, mode="a"):
    with open(save_path, mode, encoding="utf8") as f:
        for obj in save_object:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def evaluate_expression(expression):
    max_dict = {"plus": 0, "time": 0}
    def parse_expression(i):
        # Parse a term followed by either + or - followed by another term
        value, i = parse_term(i)
        while i < len(expression) and expression[i] in '+-':
            if expression[i] == '+':
                i += 1
                right_value, i = parse_term(i)
                value += right_value
                if abs(value) > abs(max_dict["plus"]):
                    max_dict["plus"] = abs(value)
            elif expression[i] == '-':
                i += 1
                right_value, i = parse_term(i)
                value -= right_value
                
        return value, i
    
    def parse_term(i):
        # Parse a factor followed by either * or / followed by another factor
        value, i = parse_factor(i)
        while i < len(expression) and expression[i] in '*/':
            if expression[i] == '*':
                i += 1
                right_value, i = parse_factor(i)
                value *= right_value
                if abs(value) > abs(max_dict["time"]):
                    max_dict["time"] = abs(value)
            elif expression[i] == '/':
                i += 1
                right_value, i = parse_factor(i)
                value /= right_value
        return value, i
    
    def parse_factor(i):
        # Parse a number or a parenthesized expression
        if expression[i] == '(':
            i += 1  # Skip '('
            value, i = parse_expression(i)
            i += 1  # Skip ')'
        else:
            start_i = i
            while i < len(expression) and (expression[i] == "." or expression[i].isdigit()):
                i += 1
            if "." in expression:
                value = float(expression[start_i:i])
            else:
                value = int(expression[start_i:i])
        return value, i
    
    # Remove whitespace from the expression
    expression = expression.replace(' ', '')
    if expression.startswith("-"):
        expression = "0" + expression
    value, _ = parse_expression(0)
    return value, max_dict

