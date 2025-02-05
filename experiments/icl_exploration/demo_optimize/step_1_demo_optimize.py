from openai import OpenAI
from tqdm import tqdm
from utils.tools import read_jsonl, write_jsonl
client = OpenAI(base_url="https://xxx", api_key="sk-xxx")
load_path = "data/biggsm/train.jsonl"
input_data = read_jsonl(load_path)
PROMPT_TEMPLATE = """[
    {
        "question": "Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?",
        "answer": "It took Leo 25 x 2 = <<25*2=50>>50 minutes to finish the second part of the assignment.\nLeo finished the first and second parts of the assignment in 25 + 50 = <<25+50=75>>75 minutes.\nHe finished the entire assignment in 60 x 2 = <<60*2=120>>120 minutes.\nTherefore, it took Leo 120 - 75 = <<120-75=45>>45 minutes to finish the third part of the assignment.\n#### 45"
    },
    {
        "question": "Liza bought 10 kilograms of butter to make cookies. She used one-half of it for chocolate chip cookies, one-fifth of it for peanut butter cookies, and one-third of the remaining butter for sugar cookies. How many kilograms of butter are left after making those three kinds of cookies?",
        "answer": "Liza used 10/2 = <<10/2=5>>5 kilograms of butter for the chocolate chip cookies.\nThen, she used 10/5 = <<10/5=2>>2 kilograms of butter for the peanut butter cookies.\nShe used 5 + 2 = <<5+2=7>>7 kilograms of butter for the chocolate and peanut butter cookies.\nSo, only 10 -7 = <<10-7=3>>3 kilograms of butter was left.\nThen, Liza used 3/3 = <<3/3=1>>1 kilograms of butter for the sugar cookies.\nTherefore, only 3-1 = <<3-1=2>>2 kilograms of butter were left.\n#### 2"
    },
    {
        "question": "Tina makes $18 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?",
        "answer": "She works 8 hours a day for $18 per hour so she makes 8*18 = $<<8*18=144>>144 per 8-hour shift\nShe works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = <<10-8=2>>2 hours of overtime\nOvertime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*0.5 = $<<18*0.5=9>>9\nHer overtime pay is 18+9 = $<<18+9=27>>27\nHer base pay is $144 per 8-hour shift and she works 5 days and makes 5 * $144 = $<<144*5=720>>720\nHer overtime pay is $27 per hour and she works 2 hours of overtime per day and makes 27*2 = $<<27*2=54>>54 in overtime pay\n2 hours of overtime pay for 5 days means she makes 54*5 = $270\nIn 5 days her base pay is $720 and she makes $270 in overtime pay so she makes $720 + $270 = $<<720+270=990>>990\n#### 990"
    },
]"""
for data in tqdm(input_data):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        stream=False,
        messages=[
            {"role": "user", "content": "[[EXAMPLE]]\n" + PROMPT_TEMPLATE+ "\n\n[[Question]]\n" + data["question"]}
        ]
        )
    res_str = completion.choices[0].message.content
    
    answer = data["answer"].replace(",", "").strip(".").split("\n#### ")[-1]
    try:
        answer = round(float(res_str), 2)
    except:
        answer = -1
    completion = client.chat.completions.create(
        model="gpt-4o",
        stream=False,
        messages=[
            {"role": "user", "content": "[[EXAMPLE]]\n" + PROMPT_TEMPLATE+ "\n\n[[Question]]\n" + data["question"] + "\n\n[[Model Prediction]]\n" + res_str + "\n\n[[Golden Answer]]\n" + answer + "\n\nPlease optimize all cases in the model '[[Example]]' based on the model's predictions and golden answer to ensure the correct understanding and reasoning of the model. Among them, all samples can be modified, and you can modify the background and value of the question. Finally, returns a JSON format example list."}
        ]
        )
    PROMPT_TEMPLATE = completion.choices[0].message.content
    write_jsonl("experiments/icl_exploration/demo_optimize/request_data/system_prompt.jsonl", [PROMPT_TEMPLATE], "a")