import os
import pandas as pd
import wandb
import argparse
import time
import sys
sys.path.append(".")
from configs.wandb_config import wandb_id
import time

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
#of the finished sweeps and the download of their results

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
# Max time
#parser.add_argument('--max_time', type=int, default=3000, help="Time after which a run is considered crashed")

# Parse the arguments
args = parser.parse_args()

df = pd.read_csv(args.filename)

#TODO YOU SHOULD ADAPT THIS COMMAND TO YOUR SITUATION
print(f"Launching {len(df)} sweeps")
for i, row in df.iterrows():
    use_gpu = args.gpu or row["use_gpu"]
    if args.cpu_only and use_gpu:
        print("Skipping sweep as it's GPU and we only want CPU")
        continue
    if args.gpu_only and not use_gpu:
        print("Skipping sweep as it's CPU and we only want GPU")
        continue
    if use_gpu:
        OAR_COMMAND = """oarsub "module load miniconda3;source activate toy_tabular;wandb agent {}/{}/{}" 
        -l gpu=1,walltime=23:00:30 -p "not cluster='graphite' 
        AND not cluster='grimani' AND not cluster='gruss'" -q production"""
        #TODO modify launch_agent_gpu.sh
        SLURM_COMMAND = "sbatch --export=wandb_id={},project={},sweep_id={} launch_benchmarks/launch_agent_gpu.sh"
    else:
        OAR_COMMAND = """oarsub "module load miniconda3;source activate toy_tabular;wandb agent {}/{}/{}" 
        -l walltime=23:00:30 -p "not cluster='graphite' 
        AND not cluster='grimani' AND not cluster='gruss'" -q production"""
        #TODO modify launch_agent.sh
        SLURM_COMMAND = "sbatch --export=wandb_id={},project={},sweep_id={} launch_benchmarks/launch_agent.sh"



    sweep = api.sweep(f"{wandb_id}/{row['project']}/{row['sweep_id']}")
    print(sweep)
    for _ in range(args.n_runs):
        print("Launching run")
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
                    print('Checking that all the runs are finished')
                    # Check that all runs are finished
                    all_finished = True
                    for run in runs:
                        if run.state == "running":
                            print(f"Run {run.name} is still running")
                            all_finished = False
                            break
                    if all_finished:
                        print("All runs are finished")
                        print("Saving results")
                        sweep_output_filename = args.output_filename.replace(".csv", "_{}.csv".format(row["sweep_id"]))
                        download_sweep(sweep, sweep_output_filename, row, max_run_per_sweep=args.max_run_per_sweep)
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
                    download_sweep(sweep, sweep_output_filename, row, max_run_per_sweep=args.max_run_per_sweep)
                    temp_filename_list.append(sweep_output_filename)
                    saved_sweeps.append(row["sweep_id"])
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
