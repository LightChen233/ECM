from utils.drawer_tool import PlotDrawer




def main():
    drawer = PlotDrawer()
    
    drawer.SAMPLE_INTERVAL=20
    drawer.MIN_SAMPLE_SIZE=10
    PATH_DICT = {
        "zero-cot": {
            "data_path": "experiments/theory-verification/cot-verification/request_data/zero-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
        },
        "no-cot": {
            "data_path": "experiments/cot-explanation/no-cot/request_data/no-cot.jsonl",
            "enable_cot": False,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/cot-explanation/no-cot/result.svg')
    drawer.draw(PATH_DICT, save_path='experiments/cot-explanation/no-cot/result-all.svg', unified_draw=True)
    
main()