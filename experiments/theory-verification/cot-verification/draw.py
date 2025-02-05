from utils.drawer_tool import PlotDrawer


def main():
    drawer = PlotDrawer()
    PATH_DICT = {
        "zero-cot": {
            "data_path": "experiments/theory-verification/cot-verification/request_data/zero-cot.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "skill",
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/cot-verification/result.svg')
    
main()