from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    PATH_DICT = {
        "skill-icl-cot": {
            "data_path": "experiments/theory-verification/unified-verification/request_data/random-icl-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
            "task_name": "math"
        },
        "roberta-icl-cot": {
            "data_path": "experiments/icl-explanation/different_embedder/request_data/roberta_result.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "roberta",
            "task_name": "math"
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/icl-explanation/meta-recognition/result.svg', print_split_cor=True)
    
main()