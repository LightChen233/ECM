import os

import torch
from tqdm import tqdm
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from utils.tools import read_jsonl, write_jsonl
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")#.to("cuda")
for i in tqdm(range(1,16)):

    data_path = fr"experiments/icl_exploration/demo_optimize/request_data/res-05/unified-{i}-tp-05.jsonl"
    if not os.path.exists(data_path):
        continue
    request_data = read_jsonl(data_path)
    data_list = []
    
    for req in tqdm(request_data):
        test_question = req['origin']['question']
        if "Question: " in req['pred'][0]['content'][0]['text']:
            train_question = req['pred'][0]['content'][0]['text'].split("Question: ")[1:-1]
        else:
            train_question = [x.split("[[Answer]]")[0] for x in req['pred'][0]['content'][0]['text'].split("[[Question]]")[1:-1]]
        test_embed = model(**tokenizer(test_question.strip(), return_tensors='pt', max_length=512))[-1][0]
        norm_test = torch.norm(test_embed)
        req["origin"]["demo_list"] = []
        for idx, tq in enumerate(train_question):
            
            encoded_input = tokenizer(tq.strip(), return_tensors='pt', max_length=512)#.to("cuda")
            train_embed = model(**encoded_input)[-1][0]
            flux = torch.dot(test_embed, train_embed) / norm_test
            req["origin"]["demo_list"].append({"index": str(idx), "flux": flux.item()})
        req["origin"]["demo_list"] = sorted(req["origin"]["demo_list"], key=lambda x: x["flux"], reverse=True)
        data_list.append(req)

    write_jsonl(data_path, data_list, "w")