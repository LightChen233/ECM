from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    PATH_DICT = {
        "icl-cot": {
            "data_path": "experiments/icl-explanation/synthetic-cot/request_data/pos_demo.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "weak-cot": {
            "data_path": "experiments/icl-explanation/synthetic-cot/request_data/neg_demo.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "zero-cot": {
            "data_path": "experiments/theory-verification/cot-verification/request_data/zero-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "reverse-cot": {
            "data_path": "experiments/theory-verification/unified-verification/request_data/reverse-icl-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/icl-explanation/synthetic-cot/result.svg', print_split_cor=True, draw_group=True)
    
main()