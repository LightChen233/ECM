import re
from textgrad.engine.openai import ChatOpenAI
import textgrad as tg
from tqdm import tqdm

from experiments.icl_exploration.demo_optimize.tool.loader import DataLoader
from experiments.icl_exploration.demo_optimize.tool.loss import MathPredictionLoss
from experiments.icl_exploration.demo_optimize.tool.model import MathPredictionModel
from utils.tools import read_jsonl, write_jsonl
from experiments.icl_exploration.demo_optimize.tool.optimizer import DemonstrationOptimizer


forward_engine = ChatOpenAI(
    model_string="gpt-3.5-turbo",
    base_url="https://xxx",
    api_key="sk-xxx")
model = MathPredictionModel(forward_engine)

backward_engine = ChatOpenAI(
    model_string="gpt-4o",
    base_url="https://xxx",
    api_key="sk-xxx")
input_data = DataLoader("data/biggsm/test.jsonl")
eval_fn = MathPredictionLoss(backward_engine)
optimizer = DemonstrationOptimizer(backward_engine, parameters=list(model.parameters()))
res_data = []
for data in tqdm(input_data):
    question, answer = data
    print(model.model.system_prompt.value)
    pred = model(question)

    
    loss = eval_fn([question, answer, pred])
    
    loss.backward(backward_engine)
    optimizer.step()
    print(model.model.system_prompt.value)
    res_data.append(model.model.system_prompt.value)
    write_jsonl("experiments/icl_exploration/demo_optimize/request_data/origin_prompt.jsonl", res_data, "a")