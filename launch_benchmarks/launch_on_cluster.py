import os
import pandas as pd
import wandb
import argparse
import time
from configs.wandb_config import wandb_id


def download_sweep(sweep, output_filename, row):
    MAX_RUNS_PER_SWEEP = 20_000
    runs_df = pd.DataFrame()

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
        dic_to_add["sweep_name"] = row["name"]
        dic_to_add["sweep_id"] = row["sweep_id"]
        dic_to_add["hp"] = "default" if "default" in row["name"] else "random"
        runs_df = runs_df.append(dic_to_add, ignore_index=True)

    runs_df.to_csv(output_filename)


api = wandb.Api()

print(f"wandb version: {wandb.__version__}")

# If you run this file with --monitor, it will launch the sweeps, then handles the killing
#of the finished sweeps and the download of their results

# Create an argument parser
parser = argparse.ArgumentParser(description='Launch runs on wandb')
# Name of the file with the sweep ids
parser.add_argument('--filename', type=str)
# Number of parallel runs per sweep
parser.add_argument('--n_runs', type=int, default=10)
# Maximum number of runs per sweep
parser.add_argument('--max_runs', type=int, default=1000)
# Name of the file to save the results
parser.add_argument('--output_filename', type=str, default="results.csv")
# Whether to launch the monitoring of the runs
parser.add_argument('--monitor', action='store_true')
# Whether to launch on OAR
parser.add_argument('--oar', action='store_true')
# Whether to use GPU
parser.add_argument('--gpu', action='store_true')
# Whether it's only default hyperparameters
parser.add_argument('--default', action='store_true')
# Time between two checks
parser.add_argument('--time', type=int, default=200) # for default hyperparameters, you want a big enough number
                                                     # due to some WandB quirk

# Parse the arguments
args = parser.parse_args()

df = pd.read_csv(args.filename)

#TODO YOU SHOULD ADAPT THIS COMMAND TO YOUR SITUATION
print(f"Launching {len(df)} sweeps")
for i, row in df.iterrows():
    use_gpu = args.gpu or row["use_gpu"]
    if use_gpu:
        OAR_COMMAND = """oarsub "module load miniconda3;source activate toy_tabular;wandb agent {}/{}/{}" 
        -l gpu=1,walltime=23:00:30 -p "not cluster='graphite' 
        AND not cluster='grimani' AND not cluster='gruss'" -q production"""
        SLURM_COMMAND = """sbatch --gres=gpu:1 --time=23:00:30 --partition=parietal,normal
        --wrap="conda activate tabular_benchmark;wandb agent {}/{}/{}" """
    else:
        OAR_COMMAND = """oarsub "module load miniconda3;source activate toy_tabular;wandb agent {}/{}/{}" 
        -l walltime=23:00:30 -p "not cluster='graphite' 
        AND not cluster='grimani' AND not cluster='gruss'" -q production"""
        SLURM_COMMAND = """sbatch --time=23:00:30 --partition=parietal,normal --exclude=marg009
        --wrap="conda activate tabular_benchmark;wandb agent {}/{}/{}" """



    sweep = api.sweep(f"{wandb_id}/{row['project']}/{row['sweep_id']}")
    print(sweep)
    for _ in range(args.n_runs):
        if not args.oar:
            if not use_gpu:
                os.system(SLURM_COMMAND.format(wandb_id, row["project"], row["sweep_id"]))
            else:
                os.system(
                    SLURM_COMMAND.format(wandb_id, row["project"], row["sweep_id"]))
        else:
            os.system(
                OAR_COMMAND.format(wandb_id, row["project"], row["sweep_id"])
                )

if args.monitor:

    saved_sweeps = []
    print("Launching monitoring")

    temp_filename_list = []

    print("Waiting...")
    time.sleep(args.time)
    print("Starting monitoring...")
    while len(saved_sweeps) < len(df):
        print("Checking...")
        api = wandb.Api()
        for i, row in df.iterrows():
            if not (row["sweep_id"] in saved_sweeps):
                sweep = api.sweep(f"{wandb_id}/{row['project']}/{row['sweep_id']}")
                runs = sweep.runs
                n = len(runs)
                print(f"{n} runs for sweep {row['sweep_id']}")
                print(sweep.state)
                # Run command line
                if sweep.state == "FINISHED":
                    print("Downloading results")
                    sweep_output_filename = args.output_filename.replace(".csv", "_{}.csv".format(row["sweep_id"]))
                    download_sweep(sweep, sweep_output_filename, row)
                    temp_filename_list.append(sweep_output_filename)
                    saved_sweeps.append(row["sweep_id"])
                if not ("default" in row[
                    "name"]) and not args.default and sweep.state == "RUNNING" and n > args.max_runs * row[
                    "n_datasets"]:
                    print("Stopping sweep")
                    os.system("wandb sweep --stop {}/{}/{}".format(wandb_id, row['project'], row['sweep_id']))
                    # Download the results
                    sweep_output_filename = args.output_filename.replace(".csv", "_{}.csv".format(row["sweep_id"]))
                    print("Downloading results")
                    download_sweep(sweep, sweep_output_filename, row)
                    temp_filename_list.append(sweep_output_filename)
                    saved_sweeps.append(row["sweep_id"])
        print("Check done")
        print("Waiting...")
        time.sleep(args.time)
    # Concatenate the results
    df = pd.concat([pd.read_csv(f) for f in temp_filename_list])
    df.to_csv(args.output_filename)

    # Delete the temporary files
    for f in temp_filename_list:
        os.remove(f)
