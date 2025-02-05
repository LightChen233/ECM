import os

from tqdm import tqdm
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from utils.tools import read_jsonl, write_jsonl
from transformers import AutoModel, AutoTokenizer

data_path = fr"experiments/ICL/skill/biggsm/test-gpt4-tp-01-1.jsonl"
# save_path = data_path.split(".jsonl")[0]+"_embed.jsonl"
save_path = "experiments/ICL/skill/biggsm/test-gpt4-tp-01_embed-1.jsonl"
request_data = read_jsonl(data_path)
data_list = []
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")#.to("cuda")
last_idx = []
if os.path.exists(save_path):
    last_idx = [x["index"] for x in read_jsonl(save_path)]
for req in tqdm(request_data):
    if "index" in last_idx:
        continue
    del req["origin"]["text"]
    data = req["origin"]
    if req["pred"][-1]["content"][0]["text"] is None:
        print(req)
    data["skill"] = req["pred"][-1]["content"][0]["text"].split("[SKILL]")[-1].strip()
    encoded_input = tokenizer(data["skill"].replace("_", " "), return_tensors='pt', max_length=512)#.to("cuda")
    output = model(**encoded_input)[-1].tolist()[0]
    data["embed"] = output
    data_list.append(data)

write_jsonl(save_path, data_list, "a")