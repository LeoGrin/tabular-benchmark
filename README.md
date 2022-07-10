# Tabular data learning benchmark

Accompanying repository for the paper *Why do tree-based models still outperform deep learning on tabular data?*

![alt text](analyses/plots/random_search_classif_numerical.jpg "Benchmark on numerical features")
![alt text](analyses/plots/random_search_regression_numerical.jpg "Benchmark on numerical features")


# Replicating the paper's results

## Training the models

You can re-run the training using WandB sweeps.

1. Copy/clone this repo on the different machines / clusters you want to use.
2. Login to WandB and create new projects.
3. Enter your projects name in `launch_config/launch_benchmarks.py` (or `launch_config/launch_xps.py`)
4. run `python launch_config/launch_benchmarks.py`
5. Run the generated sweeps using `wandb agent <USERNAME/PROJECTNAME/SWEEPID>` on the machine of your choice. 
More infos
[in the WandB doc](https://docs.wandb.ai/guides/sweeps/quickstart#4.-launch-agent-s)
6. After you've stopped the runs, download the results: `python launch_config/download_data.py`

We're planning to release a version allowing to use Benchopt instead of WandB to make it easier to run.

## Replicating the analyses / figures

All the code used to generate the analyses and figures in available in the `analyses` folder.


# Benchmarking your own algorithm

## Downloading the datasets


## Using our code


## Using our results


