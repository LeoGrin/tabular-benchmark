import numpy as np
import pandas as pd
import wandb
import time
import os

WANDB_ID = "leogrin"#INSERT WANDB ENTITY

df = pd.read_csv("sweeps/bo_nn_nouveau.csv")

api = wandb.Api()

for i, row in df.iterrows():
    sweep = api.sweep(f"{WANDB_ID}/{row['project']}/{row['sweep_id']}")
    print(sweep)
    runs = sweep.runs
    n = len(runs)
    print(n)
    # Run command line
    if n > 200 and sweep.state == "RUNNING":
    #if sweep.state == "RUNNING":
        os.system("wandb sweep --stop leogrin/{}/{}".format(row['project'], row['sweep_id']))
