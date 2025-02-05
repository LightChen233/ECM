from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    drawer.MIN_SAMPLE_SIZE = 10
    drawer.SAMPLE_INTERVAL = 30
    PATH_DICT = {
        "icl-cot": {
            "data_path": "experiments/theory-verification/more-data/med_prob/request_data/gpt35-tp-01.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "roberta",
            "task_name": "med-prob"
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/more-data/med_prob/result.svg', print_split_cor=True)
    
main()