import asyncio
from copy import deepcopy
import json
import os
import re
from tqdm import tqdm

from utils.tools import read_jsonl, write_jsonl

class RequestOutput:
    def __init__(self, load_path, auto_index=True) -> None:
        temp_list = []
        if auto_index:
            for i, temp in enumerate(read_jsonl(load_path)):
                temp["index"] = str(i)
                temp_list.append(temp)
            self.data = sorted(read_jsonl(load_path), key=lambda x: int(x["index"]))
        else:
            self.data = read_jsonl(load_path)
    
    def save(self, save_path):
        write_jsonl(save_path, self.data, mode="w")
    
    def get_last_pred_text(self, index):
        return self.data[index]["pred"][-1]["content"][0]["text"]
    
    def remove_additional_example(self):
        data_list = []
        for d in self.data:
            if isinstance(d["pred"][-1]["content"][0]["text"], list):
                d["pred"][-1]["content"][0]["text"] = d["pred"][-1]["content"][0]["text"][0].split("$<<720+270=990>>990\n#### 990\n\nQuestion: ")[-1]
            d["pred"][-1]["content"][0]["text"] = d["pred"][-1]["content"][0]["text"].split("Question:")[0].split("[Question]")[0].split("[QUESTION]")[0]
            data_list.append(d)
        self.data = data_list
            
    def get_origin_input(self, index):
         return self.data[index]["origin"]
    
    def search_by_question(self, question):
        for i, d in enumerate(self.data):
            if d["origin"]["question"] == question:
                return i
        return None
    
    def __len__(self):
        return len(self.data)

    def get_pred_answer(self, idx):
        # 
        # pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', self.get_last_pred_text(idx).replace(",", "").strip(".").split("=")[-1])]
        pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', self.get_last_pred_text(idx).replace(",", "").strip(".").split("=")[-1])]
        if len(pred_list) == 0:
            pred1 = -1
        else:
            pred1 = pred_list[-1]
        return pred1
    
    def get_json_answer(self, idx):
        last_pred_text = self.get_last_pred_text(idx).replace("[[Answer]]", "").replace("[[Question]]", "")
        if "[" not in last_pred_text:
            last_pred_text = "[" + last_pred_text
        last_pred_text = re.sub(r'(\\text)\{(.*?)\}', r'\1\\{\2\\}', last_pred_text)
        pred_list = [s for s in re.findall(r'((?<!\\)\{.*?(?<!\\)\})', last_pred_text.replace(",\n    },\n    {\n        ", "\n    },\n    {\n        ").replace(",\n    }", "\n    }").replace("\n    },\n   ]", "\n    }\n   ]"), re.M|re.I|re.S)]
        if len(pred_list) == 0:
            pred1 = [{"result": "-1"}]
        else:
            try:
                pred1 = [json.loads(temp_p) for temp_p in pred_list]
                for p_idx, p in enumerate(pred1):
                    if p["result"] != "":
                        temp_pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', p["result"].replace(",", "").strip(".").split("=")[-1])]
                        if len(temp_pred_list) == 0:
                            pred1[p_idx]["result"] = -1
                        else:
                            pred1[p_idx]["result"] = temp_pred_list[-1]
            except:
                pred1 = [{"result": "-1"}]
        return pred1

    def get_parsed_pred_answer(self, idx):
        pred_str = self.get_last_pred_text(idx)
        if "var1" not in pred_str or "<<" not in pred_str:
            pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', pred_str.replace(",", "").strip(".").split("=")[-1])]
            if len(pred_list) == 0:
                pred1 = -1
            else:
                pred1 = pred_list[-1]
            return pred1
        else:
            try:
                eqs = [s for s in re.findall(r'<<(.*?)>>', pred_str) if "=" in s]
                eqs = sorted(eqs, key=lambda x: int(x.split("=")[0].strip("var")))
                var_list = {eq.split("=")[0]: None for eq in eqs}
                for eq in eqs:
                    if "=" in eq:
                        func_str = eq.split("=")[1]
                        for var in var_list:
                            if var_list[var] is not None:
                                func_str = func_str.replace(var, str(var_list[var]))
                        # if "var" in func_str or "x" in func_str:
                        #     return -1
                        if var_list[eq.split("=")[0]] is None:
                            try:
                                var_list[eq.split("=")[0]] = eval(func_str)
                            except:
                                return -1
                    elif "var" in eq:
                        pred_str += "#### " + eq
                if "####" in pred_str:
                    var_key = pred_str.split("####")[-1].strip().strip(".").replace("<", "").replace(">", "")
                    if var_key in var_list:
                        return var_list[var_key]
                last_var = var_list[list(var_list.keys())[-1]]
                if last_var is None:
                    last_var = -1
            except:
                return -1
            return last_var
    
    def get_program_answer(self, idx):
        pred_str = self.get_last_pred_text(idx)
        if "def " not in pred_str or "```" not in pred_str:
            return self.get_pred_answer(idx)
        else:
            if "```" in pred_str:
                pred_str = pred_str.split("```")[1]
            if "while" in pred_str:
                return -1
            g = {}
            l = {}
            
            try:
                exec(pred_str.strip(), g, l)
                return l["solver"]()
            except Exception as e:
                # print(e)
                # print(pred_str)
                return -1
    
    def get_text_answer(self, idx):
        return self.get_last_pred_text(idx).split("####")[-1]
class MMRequestor:
    def __init__(self,
                 model_type="gpt",
                 model_name="gemini-pro-vision",
                 api_key="YOUR_API_KEY",
                 enable_multi_turn=False,
                 request_proxy=False) -> None:
        
        self.model_type = model_type
        self.model_name = model_name
        self.enable_multi_turn = enable_multi_turn
        self.chat = []
        if model_type == "gpt":
            from openai import AsyncOpenAI
            if request_proxy:
                client = AsyncOpenAI(api_key=api_key,
                                     timeout=600.0,
                                    base_url="https://one.ooo.cool/v1") # "https://vip.ooo.cool/v1"
            else:
                client = AsyncOpenAI(api_key=api_key)
            self.requestor = client
            
        else:
            raise ValueError("Not Supported other model besides ['gpt4', 'gemini']")
    
    async def request(self, prompts, **kargs):
        
        if self.model_type == "gpt":
            if isinstance(prompts, list):
                for prompt in prompts:
                    self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                    response = await self.requestor.chat.completions.create(
                        model=self.model_name,
                        messages=self.chat,
                        **kargs
                        )
                    self.chat.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": choice.message.content,
                        } for choice in response.choices]
                    })
            else:
                prompt = prompts
                self.chat.append({
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": prompt,
                        }],
                })
                response = await self.requestor.chat.completions.create(
                    model=self.model_name,
                    messages=self.chat,
                    **kargs
                    )
                self.chat.append({
                    "role": "assistant",
                    "content": [{
                            "type": "text",
                            "text": choice.message.content,
                        } for choice in response.choices]
                })
            res_str = deepcopy(self.chat)
            self.chat = []
            return res_str
        elif self.model_type == "gemini":
            if isinstance(prompts, list):
                for prompt in prompts:
                    self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                    try_time = 0
                    while True:
                        try:
                            response = await self.requestor.generate_content_async([x["content"][0]["text"] for x in self.chat],
                                                                                    generation_config={
                                                                                        "candidate_count": 1,
                                                                                        "max_output_tokens": 500,
                                                                                        "top_p": 1.0,
                                                                                        "temperature": 0.1
                                                                                    },
                                                                                    safety_settings={
                                                                                        "harassment": "block_none",
                                                                                        "hate": "block_none",
                                                                                        "sex": "block_none",
                                                                                        "danger": "block_none"
                                                                                    })
                            break
                        except Exception as e:
                            if "Request a higher quota limit." in str(e) or "Resource has been exhausted" in str(e):
                                asyncio.sleep(60)
                                continue
                            elif "but none were returned" in str(e) or 'list index out of range' in str(e):
                                asyncio.sleep(60)
                                try_time += 1
                                if try_time < 5:
                                    continue
                                else:
                                    raise e
                            else:
                                raise e
                    self.chat.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": response.text
                        }]
                    })
            else:
                prompt = prompts
                self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                try_time = 0
                while True:
                    try:
                        response = await self.requestor.generate_content_async([x["content"][0]["text"] for x in self.chat],
                                                                                generation_config={
                                                                                    "candidate_count": 1,
                                                                                    "max_output_tokens": 500,
                                                                                    "top_p": 1.0,
                                                                                    "temperature": 0.1
                                                                                },
                                                                                safety_settings={
                                                                                    "harassment": "block_none",
                                                                                    "hate": "block_none",
                                                                                    "sex": "block_none",
                                                                                    "danger": "block_none"
                                                                                })
                        break
                    except Exception as e:
                        if "Request a higher quota limit." in str(e) or "Resource has been exhausted" in str(e):
                            asyncio.sleep(60)
                            continue
                        elif "but none were returned" in str(e) or 'list index out of range' in str(e):
                            asyncio.sleep(60)
                            try_time += 1
                            if try_time < 5:
                                continue
                            else:
                                raise e
                        else:
                            raise e
                self.chat.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": response.text
                    }]
                })
            res_str = deepcopy(self.chat)
            self.chat = []
            return res_str
        elif self.model_type == "claude":
            if isinstance(prompts, list):
                for prompt in prompts:
                    self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                    response = await self.requestor.generate_content(
                        messages=self.chat,
                        **kargs
                        )
                    self.chat.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": response["choices"][0]["message"]["content"],
                        }]
                    })
            else:
                prompt = prompts
                self.chat.append({
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": prompt,
                        }],
                })
                response = await self.requestor.generate_content(
                    messages=self.chat,
                    **kargs
                    )
                self.chat.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": response["choices"][0]["message"]["content"],
                    }]
                })
            res_str = deepcopy(self.chat)
            self.chat = []
            return res_str
        # print(message.content)
def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf8") as f:
        f.write(json_string + "\n")

async def producer(queue, dataset, save_path, bar, create_prompt):
    if os.path.exists(save_path):
        last_request = [x["index"] for x in read_jsonl(save_path)]
    else:
        last_request = []
    for i, data in enumerate(dataset.data):
        if data["index"] in last_request:
            bar.update(1)
            continue
        prompt = create_prompt(data)
        # DSP
        print("Loaded\t\t#", data["index"])
        data.update({"index": data["index"], "text": prompt})
        await queue.put(data)
    # 所有项目都放入队列后，放入 None 表示完成
    print("Dataset Loaded.")
    await queue.put(None)


async def consumer(queue,
                   save_path,
                   bar,
                   model_type,
                   model_name,
                   api_key,
                   enable_multi_turn,
                   request_proxy,
                   return_origin,
                   model_config
                   ):
    while True:
        # 从队列中获取项目
        item = await queue.get()
        if item is None:
            print("Consumer Break")
            queue.task_done()
            # None 表示没有更多的项目
            break
        text = item["text"]
        
        
        
        try:
            print("Requesting\t\t#", item["index"])
            # await asyncio.sleep(5)  # 模拟异步操作
            requestor = MMRequestor(model_type=model_type,
                                    model_name=model_name,
                                    api_key=api_key,
                                    enable_multi_turn=enable_multi_turn,
                                    request_proxy=request_proxy)
            result = await requestor.request(
                prompts=text,
                **model_config
            )
            if return_origin:
                append_to_jsonl({"index": item["index"], "pred": result, "origin": item}, save_path)
            else:
                append_to_jsonl({"index": item["index"], "pred": result}, save_path)
            print("Saved\t\t#", item["index"])
        except Exception as e:
            print(e)
        
        # 通知队列任务已完成
        bar.update(1)
        queue.task_done()
        print(f"Queue left: {queue.qsize()}, Finished: {item['index']}")

async def request_LLM(total,
                model_type,
                model_name,
                api_key,
                enable_multi_turn,
                split=0,
                dataset=None,
                save_path = "",
                consumer_size=15,
                create_prompt_fn=None,
                request_proxy=False,
                return_origin=True,
                model_config={}):
    queue = asyncio.Queue(maxsize=60)  # 设置队列最大大小
    
    if dataset is None:
        return
    
    step = int(len(dataset.data)/total)
    dataset.data = dataset.data[step*split:min(len(dataset.data), step*(split+1))]
    bar = tqdm(total=len(dataset.data), desc=f"Total: {total} Split: {split}")
    # 创建生产者和消费者任务
    producer_task = asyncio.create_task(producer(queue, dataset, save_path, bar, create_prompt_fn))
    consumer_tasks = [asyncio.create_task(consumer(queue, save_path, bar, model_type, model_name, api_key, enable_multi_turn, request_proxy, return_origin, model_config)) for _ in range(consumer_size)]  # 创建5个消费者
    
    # 等待所有项目被处理
    await producer_task
    await queue.join()  # 等待队列被清空
    print("Cleaned Queue!")
    # 取消消费者任务
    for task in consumer_tasks:
        task.cancel()
    print("Cleaned Tasks!")
    exit()