import json
from utils.request_tool import RequestOutput
from utils.tools import read_jsonl, write_jsonl

top = "1000"
tp = "08"
path_1 = f"experiments/icl_exploration/diversity/request_data/dd-icl/res-{tp}-1/diverse-gpt35-temp-{tp}-top-{top}.jsonl"
res = RequestOutput(path_1)
top_2 = "2000"
tp_2 = "01"
path_2 = f"experiments/icl_exploration/diversity/request_data/dd-icl/res-{tp_2}-1/diverse-gpt35-temp-{tp_2}-top-{top_2}.jsonl"
res_2 = RequestOutput(path_2)

final_data = read_jsonl("experiments/icl_exploration/diversity/request_data/dd-icl/final.jsonl")
final_data_dict = {x["top"]+x["temperature"]: idx for idx, x in enumerate(final_data)}
for idx in range(len(res)):
    temp_str = res.data[idx]["pred"][-1]["content"][0]["text"]
    res_2_idx = res_2.search_by_question(res.get_origin_input(idx)["question"])
    res.data[idx]["pred"][-1]["content"][0]["text"] = res_2.data[res_2_idx]["pred"][-1]["content"][0]["text"]
    res_2.data[res_2_idx]["pred"][-1]["content"][0]["text"] = temp_str
index = final_data_dict[top+tp]
index_2 = final_data_dict[top_2+tp_2]
temp = final_data[index]["acc"]
final_data[index]["acc"] = final_data[index_2]["acc"]
final_data[index_2]["acc"] = temp
# write_jsonl("experiments/icl_exploration/diversity/request_data/dd-icl/final.jsonl",final_data, "w")
res.save(path_1)
res_2.save(path_2)