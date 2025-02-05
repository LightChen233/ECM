from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    drawer.MIN_SAMPLE_SIZE = 0
    drawer.SAMPLE_INTERVAL = 10000
    drawer.MAX_POWER = 10000
    PATH_DICT = {
        "random-icl-1": {
            "data_path": "experiments/icl_exploration/diversity/request_data/random-icl/gpt35-tp-01.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "random-icl-2": {
            "data_path": "experiments/icl_exploration/diversity/request_data/random-icl/gpt35-tp-02.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "random-icl-3": {
            "data_path": "experiments/icl_exploration/diversity/request_data/random-icl/gpt35-tp-08.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "static-icl-1": {
            "data_path": "experiments/icl_exploration/diversity/request_data/ds-icl/gpt35-tp-01.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "static-icl-2": {
            "data_path": "experiments/icl_exploration/diversity/request_data/ds-icl/gpt35-tp-02.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "static-icl-3": {
            "data_path": "experiments/icl_exploration/diversity/request_data/ds-icl/gpt35-tp-08.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "dynamic-icl-1": {
            "data_path": "experiments/icl_exploration/diversity/request_data/sd-icl/gpt35-tp-01.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "dynamic-icl-2": {
            "data_path": "experiments/icl_exploration/diversity/request_data/sd-icl/gpt35-tp-02.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "dynamic-icl-3": {
            "data_path": "experiments/icl_exploration/diversity/request_data/sd-icl/gpt35-tp-08.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
    }
    drawer.draw(PATH_DICT, save_path='', print_split_cor=True, draw_flag=False)
    
main()