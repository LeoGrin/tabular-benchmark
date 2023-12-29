import os

import numpy as np
import pandas as pd
import wandb
import argparse
import time
import subprocess
import sys

sys.path.append(".")
from configs.wandb_config import wandb_id
import time

def get_gpu_usage():
    # Define the command to be executed
    username = "$USER"
    command = 'squeue -O "username,gres:100" -u {} | awk -F: \'/gpu/ {{sum += $2}} END {{print sum}}\''.format(username)

    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)

    # Get the output and convert it to an integer
    gpu_count_str = result.stdout.strip()
    gpu_count = int(gpu_count_str) if gpu_count_str.isdigit() else 0

    return gpu_count

def download_sweep(sweep, output_filename, row, max_run_per_sweep=20000):
    MAX_RUNS_PER_SWEEP = max_run_per_sweep
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
# of the finished sweeps and the download of their results

# Create an argument parser
parser = argparse.ArgumentParser(description='Launch runs on wandb')
# Name of the file with the sweep ids
parser.add_argument('--filename', type=str)
# Number of parallel runs per sweep
parser.add_argument('--n_runs', type=int, default=10)
# Maximum number of runs per dataset
parser.add_argument('--max_runs', type=int, default=1000)
# Name of the file to save the results
parser.add_argument('--output_filename', type=str, default="results.csv")
# Whether to launch the monitoring of the runs
parser.add_argument('--monitor', action='store_true')
# Whether to launch on OAR
parser.add_argument('--oar', action='store_true')
# Whether to use GPU
parser.add_argument('--gpu', action='store_true')
# Maximum number of gpu used by the user
parser.add_argument('--max_gpus', type=int, default=10)
# Whether to only launch the sweeps running on CPU
parser.add_argument('--cpu_only', action='store_true')
# Whether to only launch the sweeps running on GPU
parser.add_argument('--gpu_only', action='store_true')
# Whether it's only default hyperparameters
parser.add_argument('--default', action='store_true')
# Time between two checks
parser.add_argument('--time', type=int, default=200)
# Max number of runs per sweep (to speed up the download)
parser.add_argument('--max_run_per_sweep', type=int, default=20000)
# Queue to use
parser.add_argument('--oar_queue', type=str, default="default")
# Max time
# parser.add_argument('--max_time', type=int, default=3000, help="Time after which a run is considered crashed")

# Parse the arguments
args = parser.parse_args()

if args.oar:
    print("Using OAR")
    if args.oar_queue == "default":
        args.oar_queue = "production"
    print(f"Using queue {args.oar_queue}")
else:
    print("Using SLURM")

df = pd.read_csv(args.filename)

# TODO YOU SHOULD ADAPT THIS COMMAND TO YOUR SITUATION
print(f"Launching {len(df)} sweeps")
for i, row in df.iterrows():
    use_gpu = args.gpu or row["use_gpu"]
    if use_gpu:
        print("Using GPU")
        # check that we are not using more gpus than 
        # wait for the job to be launched
        time.sleep(10)
        gpu_count = get_gpu_usage()
        print("Total gpu usage", gpu_count)
        while gpu_count + 1 > args.max_gpus:
            print(f"Using currently {gpu_count} gpus")
            print(f"Max gpu usage: {args.max_gpus}")
            print("Waiting")
            time.sleep(60)
            gpu_count = get_gpu_usage()


    if args.cpu_only and use_gpu:
        print("Skipping sweep as it's GPU and we only want CPU")
        continue
    if args.gpu_only and not use_gpu:
        print("Skipping sweep as it's CPU and we only want GPU")
        continue
    try:
        sweep = api.sweep(f"{wandb_id}/{row['project']}/{row['sweep_id']}")
        if sweep.state == "FINISHED" or sweep.state == "CANCELLED":
            print(f"Sweep {row['sweep_id']} already finished, skipping")
            continue
    except:
        # If the sweep doesn't exist, we launch it
        pass
    if use_gpu:
        OAR_COMMAND = """oarsub "module load miniconda3;source activate toy_tabular;wandb agent {}/{}/{}"  -l gpu=1,walltime=23:00:30 -p "not cluster='graphite' AND not cluster='grimani' AND not cluster='gruss'" -q {}"""
        # TODO modify launch_agent_gpu.sh
        SLURM_COMMAND = "sbatch --export=wandb_id={},project={},sweep_id={} launch_benchmarks/launch_agent_gpu.sh"
    else:
        OAR_COMMAND = """oarsub "module load miniconda3;source activate toy_tabular;wandb agent {}/{}/{}" 
        -l walltime=23:00:30 -p "not cluster='graphite' 
        AND not cluster='grimani' AND not cluster='gruss'" -q production"""
        # TODO modify launch_agent.sh
        SLURM_COMMAND = "sbatch --export=wandb_id={},project={},sweep_id={} launch_benchmarks/launch_agent.sh"

    try:
        sweep = api.sweep(f"{wandb_id}/{row['project']}/{row['sweep_id']}")
        print(sweep)
    except:
        print(f"sweep {row['sweep_id']} does not exist, skipping")
        continue
    if ("default" in row["name"]):
        n_runs_to_launch = np.min([args.n_runs, row["n_datasets"]])
    else:
        n_runs_to_launch = args.n_runs
    for _ in range(n_runs_to_launch):
        print("Launching run")
        if not args.oar:
            command = SLURM_COMMAND.format(wandb_id, row["project"], row["sweep_id"])
        else:
            command = OAR_COMMAND.format(wandb_id, row["project"], row["sweep_id"], args.oar_queue)
        print(command)
        os.system(
            command
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
            print(row["sweep_id"])
            if not (row["sweep_id"] in saved_sweeps):
                try:
                    sweep = api.sweep(f"{wandb_id}/{row['project']}/{row['sweep_id']}")
                # except if the sweep doesn't exist
                except:
                    print(f"Sweep {row['sweep_id']} doesn't exist")
                    saved_sweeps.append(row["sweep_id"])
                    continue
                sweep_output_filename = args.output_filename.replace(".csv", "_{}.csv".format(row["sweep_id"]))
                if os.path.exists(sweep_output_filename):
                    print("file already exists")
                    try:
                        # Check that the file contains the same number of runs as the sweep
                        saved_runs = pd.read_csv(sweep_output_filename)
                        if len(saved_runs) == len(sweep.runs):
                            print("file already contains all runs")
                            saved_sweeps.append(row["sweep_id"])
                            temp_filename_list.append(sweep_output_filename)
                            continue
                        else:
                            print("file doesn't contain all runs")
                    except:
                        print("Error when reading file")
                        pass
                runs = sweep.runs
                n = len(runs)
                print(f"{n} runs for sweep {row['sweep_id']}")
                print(sweep.state)
                # Run command line
                # Already stopped sweeps or sweeps with default hyperparameters
                if sweep.state == "FINISHED" or sweep.state == "CANCELED":
                    # WandB says that a sweep is finished when there are no more runs to launch
                    # but doesn't wait for the runs to finish
                    print('Checking that all the runs are finished')
                    # Check that all runs are finished
                    all_finished = True
                    try:
                        for run in runs:
                            if run.state == "running":
                                print(f"Run {run.name} is still running")
                                all_finished = False
                                break
                    except:
                        print("Error when checking runs")
                        all_finished = False
                    if all_finished:
                        print("All runs are finished (or crashed)")
                        print("Saving results")
                        download_sweep(sweep, sweep_output_filename, row, max_run_per_sweep=args.max_run_per_sweep)
                        temp_filename_list.append(sweep_output_filename)
                        saved_sweeps.append(row["sweep_id"])
                if "n_run_per_dataset" in row.keys():
                    n_run_per_dataset = row["n_run_per_dataset"]
                else:
                    n_run_per_dataset = 1
                print(f"we want n_run_per_dataset = {n_run_per_dataset}")
                if not ("default" in row[
                    "name"]) and not args.default and sweep.state == "RUNNING" and n > args.max_runs * row[
                    "n_datasets"] * n_run_per_dataset:
                    print("Sweep seems to be done, checking that there are enough runs for each dataset")
                    # Check that there are enough runs for each dataset
                    n_finished_runs_per_dataset = {}
                    try:
                        for run in runs:
                            if run.state == "finished":
                                # Check that there is a mean_test_score value logged
                                if "mean_test_score" in run.summary and not pd.isnull(run.summary["mean_test_score"]):
                                    if "data__keyword" in run.config.keys():
                                        dataset = run.config["data__keyword"]
                                        if dataset in n_finished_runs_per_dataset:
                                            n_finished_runs_per_dataset[dataset] += 1
                                        else:
                                            n_finished_runs_per_dataset[dataset] = 1
                    except:
                        print("Error when checking runs")
                        continue
                    print(n_finished_runs_per_dataset)
                    if np.all([n_finished_runs_per_dataset[dataset] >= args.max_runs * n_run_per_dataset for dataset in
                               n_finished_runs_per_dataset]):
                        if len(n_finished_runs_per_dataset) != row["n_datasets"]:
                            print("WARNING: some datasets are missing")
                            print(f"Expected {row['n_datasets']} datasets, got {len(n_finished_runs_per_dataset)}")
                            print("Stopping anyway")
                        print("Stopping sweep")
                        os.system("wandb sweep --stop {}/{}/{}".format(wandb_id, row['project'], row['sweep_id']))
                        # Download the results
                        print("Downloading results")
                        download_sweep(sweep, sweep_output_filename, row, max_run_per_sweep=args.max_run_per_sweep)
                        temp_filename_list.append(sweep_output_filename)
                        saved_sweeps.append(row["sweep_id"])
                    else:
                        print("Not enough runs for each dataset")
        print("Check done")
        print("Waiting...")
        time.sleep(args.time)

    print("All sweeps are finished !")
    # Concatenate the results
    df = pd.concat([pd.read_csv(f) for f in temp_filename_list])
    # Print number of runs without mean_test_score
    print("Number of runs without mean_test_score (crashed): {}".format(len(df[df["mean_test_score"].isna()])))
    # Print nan mean_test_score per model
    print("Number of runs per model without mean_test_score (crashed):")
    print(df[df["mean_test_score"].isna()].groupby("model_name").size())
    df.to_csv(args.output_filename)
    print("Results saved in {}".format(args.output_filename))
    print("Cleaning temporary files...")
    # Delete the temporary files
    for f in temp_filename_list:
        os.remove(f)
