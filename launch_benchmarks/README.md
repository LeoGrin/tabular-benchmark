# Run the benchmark

You should run `launch_benchmarks.py`, which will give you a csv file
with all the sweeps to run. For running the experiments, run `launch_xp.py`.

Then you can either run each sweep with `wandb agent <USERNAME/PROJECTNAME/SWEEPID>`, 
or adapt the `launch_on_cluster.py` script to launch all the sweeps on a SLURM
or OAR cluster.