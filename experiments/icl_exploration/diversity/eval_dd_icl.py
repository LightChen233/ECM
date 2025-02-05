import os
from utils.drawer_tool import PlotDrawer
from utils.tools import write_jsonl




def main():
    drawer = PlotDrawer()
    drawer.MIN_SAMPLE_SIZE = 0
    drawer.SAMPLE_INTERVAL = 10000
    drawer.MAX_POWER = 100000
    PATH_DICT = PATH_DICT = {
        f"icl-cot-temp-{temp}-top-{i}": {
            "data_path": f"experiments/icl_exploration/diversity/request_data/dd-icl/res-{temp}-1/diverse-gpt35-temp-{temp}-top-{i}.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        }
        for i in ["5", "10", "20", "50", "100", "200", "500", "1000", "2000"] for temp in ["02", "01", "08"] if os.path.exists(f"experiments/icl_exploration/diversity/request_data/dd-icl/res-{temp}-1/diverse-gpt35-temp-{temp}-top-{i}.jsonl")
    }
    middle_res = drawer.draw(PATH_DICT, save_path='', print_split_cor=True, draw_flag=False)
    res_list = []
    for key in middle_res:
        res_list.append({
            "acc": middle_res[key]["C"],
            "key": "icl",
            "p": middle_res[key]["P"],
            "top": key.split("-")[-1],
            "temperature": key.split("-")[3],
        })
    write_jsonl("experiments/icl_exploration/diversity/request_data/dd-icl/final.jsonl", res_list, "w")
    
main()