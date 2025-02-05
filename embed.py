import os

from tqdm import tqdm
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from utils.tools import read_jsonl, write_jsonl
from transformers import AutoModel, AutoTokenizer
import torch
SPLIT = "train"
DATA_INDEX = "gsm8k"
data_path = fr"data/{DATA_INDEX}/{SPLIT}.jsonl"
request_data = read_jsonl(data_path)
data_list = []
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")# BAAI/bge-m3 , roberta-base, bert-base-uncased
for req in tqdm(request_data):
    encoded_input = tokenizer(req["question"].replace("_", " "), return_tensors='pt', max_length=512)#.to("cuda")
    output = model(**encoded_input)[-1].tolist()[0]
    req["embed"] = output
    data_list.append(req)

write_jsonl(fr"semantic/roberta_embed/{DATA_INDEX}_{SPLIT}.jsonl", data_list, "w")