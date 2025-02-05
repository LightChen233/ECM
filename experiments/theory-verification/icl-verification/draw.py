from utils.drawer_tool import PlotDrawer


def main():
    drawer = PlotDrawer()
    drawer.SAMPLE_INTERVAL = 40
    PATH_DICT = {
        "bge-chebyshev-icl-cot": {
            "data_path": "experiments/theory-verification/icl-verification/request_data/chebyshev_distance.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bge",
        },
        "bge-cosine-icl-cot": {
            "data_path": "experiments/theory-verification/icl-verification/request_data/cosine_distance.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bge",
        },
        "bge-projection_length-cot": {
            "data_path": "experiments/icl-explanation/different_embedder/request_data/bge_result.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bge",
        },
        "bge-euclidean-icl-cot": {
            "data_path": "experiments/theory-verification/icl-verification/request_data/euclidean_distance.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bge",
        },
        "bge-seuclidean-icl-cot": {
            "data_path": "experiments/theory-verification/icl-verification/request_data/seuclidean_distance.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bge",
        },
        "bge-manhattan-icl-cot": {
            "data_path": "experiments/theory-verification/icl-verification/request_data/manhattan_distance.jsonl",
            "enable_cot": True,
            "reason_model": "gpt-3.5-turbo",
            "represent_model": "bge",
            "distance_model": "manhattan",
        },
    }
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/icl-verification/result.svg', print_split_cor=True, specific_color=False)
    
main()