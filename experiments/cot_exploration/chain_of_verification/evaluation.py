from tqdm import tqdm
from utils.tools import read_jsonl

from experiments.cot_exploration.chain_of_verification.valid_tool import judge_pass
for step in range(1, 6):
    for temperature in ["00", "01", "02", "05", "08"]:
        data_list = read_jsonl(f"experiments/cot_exploration/chain_of_verification/request_data/verify_step_{step}/gpt-4o-5-tp-{temperature}.jsonl")
        Correct = 0
        Total = 0
        for data in tqdm(data_list):
            print(f"=========Processing: {data['index']}==========")
            test_list = []
            for test_key in ["public_tests", "private_tests", "generated_tests"]:
                for x, y in zip(data['origin'][test_key]["input"], data['origin'][test_key]["output"]):
                    test_list.append((x, y))
            flag = False
            data["correct"] = []
            for code_idx, con in enumerate(data['pred'][1]['content']):
                # if data["index"] in [49, 157]:
                #     continue
                temp_flag = True
                for tdx, test_case in enumerate(test_list):
                    if not judge_pass(con['text'], test_case[0], test_case[1]):
                        temp_flag = False
                        break
                if temp_flag:
                    flag = True
                    data["correct"].append(code_idx)

            if flag:
                Correct += 1
                print(f"Correct: {data['index']}")
            Total += 1
        print(Correct/Total)
exit()