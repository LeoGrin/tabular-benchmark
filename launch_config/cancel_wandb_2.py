import numpy as np
import pandas as pd
import wandb
import time
import os

WANDB_ID = "leogrin"#INSERT WANDB ENTITY

df = pd.read_csv("sweeps/new_trees.csv")

api = wandb.Api()

for i, row in df.iterrows():
    sweep = api.sweep(f"{WANDB_ID}/{row['project']}/{row['sweep_id']}")
    #if "rf" in row["name"]:
    #    if sweep.state == "RUNNING":
    #        os.system("wandb sweep --stop leogrin/{}/{}".format(row['project'], row['sweep_id']))
    if sweep.state == "RUNNING":
        print(sweep)
        runs = sweep.runs
        n = len(runs)
        print(n)
        # Run command line
        if (n > 8000 or (n > 2000 and "large" in row["name"]))  and sweep.state == "RUNNING":#n > 2000 and "large" in row['name']:
            os.system("wandb sweep --stop leogrin/{}/{}".format(row['project'], row['sweep_id']))
