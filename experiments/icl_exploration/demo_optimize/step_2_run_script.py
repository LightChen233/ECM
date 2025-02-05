import argparse
import os

for temp in [0.1, 0.5, 0.8]:
    for i in range(16):
        os.system(f"python experiments/icl_exploration/demo_optimize/step_2_apply_optimize_prompt.py --step {i} --temperature {temp}")