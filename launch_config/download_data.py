import numpy as np
import pandas as pd
import wandb
import time

api = wandb.Api()
WANDB_ID = "leogrin"  # set to your entity and project

sweep_id_filename = "sweeps/xps_sweeps.csv"
#weep_id_filename = "sweeps/benchmark_sweeps.csv"

output_filename = "results/xps_results.csv"
#output_filename = "results/benchmark_results.csv"

df = pd.read_csv(sweep_id_filename)
sweeps = iter([api.sweep(f"{WANDB_ID}/{row['project']}/{row['sweep_id']}") for i, row in df.iterrows()])

MAX_RUNS_PER_SWEEP = np.Inf  # replace for speed

runs_df = pd.DataFrame()

while True:
    for _ in range(100):
        try:
            sweep = next(sweeps, -1)
            break
        except:
            print("error, retrying in 10 sec")
            time.sleep(10)
    if sweep == -1:
        break
    print("sweep: {}".format(sweep))
    runs = sweep.runs
    n = len(runs)
    i = 0
    runs = iter(runs)
    while True:
        for _ in range(20):
            try:
                run = next(runs, -1)
                break
            except:
                print("error, retrying in 10 sec")
                time.sleep(10)
        if run == -1:
            break
        if i > MAX_RUNS_PER_SWEEP:
            break
        if i % 100 == 0:
            print(f"{i}/{n}")
        i += 1
        config = {k: v for k, v in run.config.items()
                  if not k.startswith('_')}
        summary = run.summary
        dic_to_add = {**config, **summary}
        runs_df = runs_df.append(dic_to_add, ignore_index=True)

runs_df.to_csv(output_filename)
