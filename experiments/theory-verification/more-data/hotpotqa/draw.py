from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    drawer.MIN_SAMPLE_SIZE = 7
    drawer.SAMPLE_INTERVAL = 50
    PATH_DICT = {
        "icl-cot": {
            "data_path": "experiments/theory-verification/more-data/hotpotqa/request_data/gpt35-tp-01.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "roberta",
            "task_name": "multihopqa"
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/more-data/hotpotqa/result.svg', print_split_cor=True)
    
main()