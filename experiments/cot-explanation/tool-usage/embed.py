import os

import torch
from tqdm import tqdm
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from utils.tools import read_jsonl, write_jsonl
from transformers import AutoModel, AutoTokenizer
data_path = fr"experiments/tool-usage/pot/unified-tp-00-1.jsonl"
request_data = read_jsonl(data_path)
data_list = []
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")#.to("cuda")
train_list = [
    "Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?",
    "Liza bought 10 kilograms of butter to make cookies. She used one-half of it for chocolate chip cookies, one-fifth of it for peanut butter cookies, and one-third of the remaining butter for sugar cookies. How many kilograms of butter are left after making those three kinds of cookies?",
    "Tina makes $18 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?"
]
train_embed_list = []
for q in tqdm(train_list):
    train_embed_list.append(model(**tokenizer(q.strip(), return_tensors='pt', max_length=512))[-1][0])

for req in tqdm(request_data):
    test_question = req['origin']['question']
    test_embed = model(**tokenizer(test_question.strip(), return_tensors='pt', max_length=512))[-1][0]
    norm_test = torch.norm(test_embed)
    req["origin"]["demo_list"] = []
    for idx, train_embed in enumerate(train_embed_list):
        flux = torch.dot(test_embed, train_embed) / norm_test
        req["origin"]["demo_list"].append({"index": str(idx), "flux": flux.item()})
    req["origin"]["demo_list"] = sorted(req["origin"]["demo_list"], key=lambda x: x["flux"], reverse=True)
    data_list.append(req)

write_jsonl(data_path, data_list, "w")