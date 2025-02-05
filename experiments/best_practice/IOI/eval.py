from io import StringIO
import json
import os
import sys
from tqdm import tqdm


# problem_name = "sphinx"
# problem_name = "tree"
# problem_name = "message"
# problem_name = "mosaic"
problem_name = "nile"

load_path = f"experiments/cot_exploration/code_contest/data/{problem_name}/tests"

tasks_path = f"experiments/cot_exploration/code_contest/data/{problem_name}/subtasks"
if problem_name == "sphinx":
    from experiments.cot_exploration.code_contest.request_data.sphinx import main
elif problem_name == "message":
    from experiments.cot_exploration.code_contest.request_data.message import main
elif problem_name == "tree":
    from experiments.cot_exploration.code_contest.request_data.tree import main
elif problem_name == "mosaic":
    from experiments.cot_exploration.code_contest.request_data.mosaic import main
elif problem_name == "hieroglyphs":
    from experiments.cot_exploration.code_contest.request_data.hieroglyphs import main
elif problem_name == "nile":
    from experiments.cot_exploration.code_contest.request_data.nile import main
files = sorted(list(set([x.split(".")[0] for x in os.listdir(load_path)])))
task_list = [json.load(open(os.path.join(tasks_path, x))) for x in os.listdir(tasks_path)]
correct_list = []
origin_sys_out = sys.stdout
for f_idx, file in enumerate(tqdm(files)):
    file_path = os.path.join(load_path, file)
    with open(file_path+".in", "r", encoding="utf8") as f:
        if problem_name in ["tree", "mosaic", "hieroglyphs", "nile"]:
            input_str = "".join(f.readlines()[1:])
        else:
            input_str = "".join(f.readlines())
    
    sys.stdin = StringIO(input_str.strip("\n")+"\n")
    sys.stdout = StringIO()
    main()
    actual_output = sys.stdout.getvalue()

    with open(file_path+".out", "r", encoding="utf8") as f:
        if problem_name in ["tree", "mosaic", "hieroglyphs", "nile"]:
            golden_output = "".join(f.readlines()[2:])
        else:
            golden_output = "".join(f.readlines())
    # actual_output = golden_output
    if actual_output != golden_output:
        sys.stdout = origin_sys_out
        # print("Pred: ", actual_output)
        # print("Gold: ", golden_output)
    else:
        sys.stdout = origin_sys_out
        print("Correct!", file)
        correct_list.append(file)

score = 0
for task in task_list:
    flag = True
    for case in task["testcases"]:
        if case not in correct_list:
            flag = False
            break
    if flag:
        score += task["score"]
sys.stdout = origin_sys_out
print(score)