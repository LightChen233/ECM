from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    drawer.MIN_SAMPLE_SIZE = 7
    drawer.SAMPLE_INTERVAL = 1
    PATH_DICT = {
        "task_5_plan_generalization": {
            "data_path": "experiments/theory-verification/more-data/blockworlds/request_data/task_5_plan_generalization.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "roberta",
            "task_name": "robotic-planning"
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/more-data/blockworlds/result.svg', print_split_cor=True)
    
main()