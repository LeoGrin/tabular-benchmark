#!/bin/bash
#SBATCH --job-name=tabular_benchmark_gpu
#SBATCH --partition=parietal,normal,gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=4000
bash -l -c "conda activate test;wandb agent $wandb_id/$project/$sweep_id"