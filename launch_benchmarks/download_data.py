import numpy as np
import pandas as pd
import wandb
import time

WANDB_ID = "leogrin"#INSERT WANDB ENTITY

api = wandb.Api()


sweep_id_filename = "launch_benchmarks/sweeps/resnet_several_random.csv"
#weep_id_filename = "sweeps/benchmark_sweeps.csv"

output_filename = "launch_benchmarks/results/resnet_several_random.csv"
#output_filename = "results/benchmark_results.csv"

df = pd.read_csv(sweep_id_filename)
sweeps = iter([api.sweep(f"{WANDB_ID}/{row['project']}/{row['sweep_id']}") for i, row in df.iterrows()])
#sweeps = iter([api.sweep("leogrin/nouveau_trees_1/vyjjbkm7")])

MAX_RUNS_PER_SWEEP = 6000#9000#np.Inf  # replace for speed

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
    #print("skipping last runs")
    #for _ in range(n - MAX_RUNS_PER_SWEEP):
    #    next(runs)
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
        run_name = run.name
        step = int(run.name.split("-")[-1])
        config["sweep"] = sweep.name
        config["step"] = step
        summary = run.summary
        dic_to_add = {**config, **summary}
        runs_df = runs_df.append(dic_to_add, ignore_index=True)

runs_df.to_csv(output_filename)
