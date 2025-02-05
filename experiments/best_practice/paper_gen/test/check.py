import os
import pandas as pd


res_list = pd.read_csv('experiments/best_practice/paper_gen/res.csv', delimiter=",")
import csv
res_list = []
with open('experiments/best_practice/paper_gen/res.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    res_list = [row for row in reader]
for res in list(res_list):
    temp_str = "optimized" if res['type'] == "aug" else "origin"
    if not os.path.exists(f"experiments/best_practice/paper_gen/request_data/{res['domain']}/{temp_str}/generated_idea/{res['index']}.pdf"):
        print({res['domain']}, temp_str, {res['index']})