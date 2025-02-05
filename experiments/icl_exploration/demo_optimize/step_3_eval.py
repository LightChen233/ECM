from utils.drawer_tool import PlotDrawer
from utils.tools import write_jsonl

def main():
    drawer = PlotDrawer()
    drawer.SAMPLE_INTERVAL = 40
    drawer.MAX_POWER = 3000
    PATH_DICT = {f"cot-{i}-{key}": {
        "data_path": f"experiments/icl_exploration/demo_optimize/request_data/{key}/unified-{i}-tp-{key.split('-')[-1]}.jsonl",
        "enable_cot": True,
        "reason_model": "gpt-3.5-turbo",
        "represent_model": "roberta",
    } for i in range(1,16) for key in ["origin-01", "origin-05", "origin-08", "res-01", "res-05", "res-08", ]}
    res_dict = drawer.draw(PATH_DICT, save_path='', draw_flag=False)
    res_list = []
    for key in res_dict:
        res_list.append({
            "acc": res_dict[key]["C"],
            "key": "icl",
            "p": res_dict[key]["P"],
            "iter": key.split("-")[1],
            "temperature": key.split("-")[-1],
            "type": key.split("-")[-2]
        })
    write_jsonl("experiments/icl_exploration/demo_optimize/request_data/final.jsonl", res_list, "w")
    
main()