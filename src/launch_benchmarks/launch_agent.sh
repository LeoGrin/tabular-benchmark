#!/bin/bash
#SBATCH --job-name=tabular_benchmark
#SBATCH --partition=parietal,normal
#SBATCH --nodes=1
#SBATCH --time=4000
#SBATCH --exclude=margpu009
bash -l -c "conda activate test;wandb agent $wandb_id/$project/$sweep_id"