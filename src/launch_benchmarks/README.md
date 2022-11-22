# Run the benchmark

1. Login to WandB and add your wandb id to `src/configs/wandb_config.py`
2. Move into `src`
3. run `python launch_config/launch_benchmarks.py`. This will create a csv with the wandb sweep to run. 
4. (Long?) You can run each sweep by running `wandb agent <USERNAME/PROJECTNAME/SWEEPID>` in `src`. More infos
[in the WandB doc](https://docs.wandb.ai/guides/sweeps/quickstart#4.-launch-agent-s).
5. If your using a cluster, run `launch_benchmarks/launch_on_cluster.py --filename NAME_OF_THE_CSV_FILE --output_filename FILENAME --n_runs NUMBER_OF_PARALLEL_RUNS_PER_SWEEP --max_runs MAX_NUMBER_OF_RUN_PER_DATASET --monitor`. 
You'll need to adapt the 
script to your cluster (see the TODO comments in the script). This will automatically launch the sweeps on the cluster
and download the results when they are done.