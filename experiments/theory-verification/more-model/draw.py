from utils.drawer_tool import PlotDrawer


MODEL_LIST = [
    "gemini-1.5-pro", "gemini-1.5-flash",
    "glm-3-turbo", "glm-4",
    "gpt3.5", "o1-mini", "o1-preview",
    "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct",
    "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-sonnet-20240229"
]

def main():
    
    drawer = PlotDrawer()
    drawer.MAX_POWER = 1750
    PATH_DICT = {
        x: {
            "data_path": f"experiments/theory-verification/more-model/{x}/test-5-tp-01.jsonl",
            "enable_cot": True,
            "reason_model": x,
            "represent_model": "skill",
        } for x in MODEL_LIST
    }
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/more-model/result.svg', specific_color=False, unified_draw=True)
    drawer.draw(PATH_DICT, save_path='experiments/theory-verification/more-model/result-color.svg', specific_color=False, unified_draw=False)
    
main()