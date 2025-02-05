import json
from utils.request_tool import RequestOutput
from utils.tools import read_jsonl, write_jsonl

tp = "01"
path_1 = f"experiments/icl_exploration/demo_optimize/request_data/res-{tp}/unified-1-tp-{tp}.jsonl"
res = RequestOutput(path_1)
path_2 = f"experiments/icl_exploration/demo_optimize/request_data/res-{tp}/unified-14-tp-{tp}.jsonl"
res_2 = RequestOutput(path_2)

for idx in range(len(res)):
    temp_str = res.data[idx]["pred"][-1]["content"][0]["text"]
    res_2_idx = res_2.search_by_question(res.get_origin_input(idx)["question"])
    if res_2_idx is None:
        continue
    res.data[idx]["pred"][-1]["content"][0]["text"] = res_2.data[res_2_idx]["pred"][-1]["content"][0]["text"]
    res_2.data[res_2_idx]["pred"][-1]["content"][0]["text"] = temp_str

res.save(path_1)
res_2.save(path_2)