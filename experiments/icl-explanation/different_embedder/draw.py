from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    drawer.SAMPLE_INTERVAL = 40
    PATH_DICT = {
        "bert-icl-cot": {
            "data_path": "experiments/icl-explanation/different_embedder/request_data/bert_result.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bert",
            "task_name": "math"
        },
        "bge-icl-cot": {
            "data_path": "experiments/icl-explanation/different_embedder/request_data/bge_result.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bge",
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
    drawer.draw(PATH_DICT, save_path='experiments/icl-explanation/different_embedder/result.svg', print_split_cor=True)
    
main()