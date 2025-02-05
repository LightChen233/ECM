import os

for temp in [0.1, 0.2, 0.8]:
    for i in ["5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000"]:
        os.system(f"python experiments/icl_exploration/diversity/request.py --top {i} --temperature {temp}")