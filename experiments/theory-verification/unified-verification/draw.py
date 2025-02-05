from utils.drawer_tool import PlotDrawer

def main():
    drawer = PlotDrawer()
    PATH_DICT = {
        "icl-cot": {
            "data_path": "experiments/theory-verification/unified-verification/request_data/retrieval-icl-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
        },
        "reverse-icl-cot": {
            "data_path": "experiments/theory-verification/unified-verification/request_data/reverse-icl-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
        },
        "random-icl-cot": {
            "data_path": "experiments/theory-verification/unified-verification/request_data/random-icl-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/unified-verification/result.svg',print_split_cor=True, unified_draw=True)
    
main()