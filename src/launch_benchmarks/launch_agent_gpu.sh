#!/bin/bash
#SBATCH --job-name=tabular_benchmark_gpu
#SBATCH --partition=parietal,normal,gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=4000
#SBATCH --exclude=margpu009
bash -l -c "micromamba activate test;wandb agent $wandb_id/$project/$sweep_id"