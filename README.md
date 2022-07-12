# Tabular data learning benchmark

Accompanying repository for the paper *Why do tree-based models still outperform deep learning on tabular data?*

![alt text](analyses/plots/random_search_classif_numerical.jpg "Benchmark on numerical features")

# Replicating the paper's results

## Downloading the datasets

To download these datasets, simply run `python data/download_data.py`.

## Training the models

You can re-run the training using WandB sweeps.

1. Copy / clone this repo on the different machines / clusters you want to use.
2. Login to WandB and create new projects.
3. Enter your projects name in `launch_config/launch_benchmarks.py` (or `launch_config/launch_xps.py`)
4. run `python launch_config/launch_benchmarks.py`
5. Run the generated sweeps using `wandb agent <USERNAME/PROJECTNAME/SWEEPID>` on the machine of your choice. 
More infos
[in the WandB doc](https://docs.wandb.ai/guides/sweeps/quickstart#4.-launch-agent-s)
6. After you've stopped the runs, download the results: `python launch_config/download_data.py`, after entering your wandb
login in `launch_config/download_data.py`.

We're planning to release a version allowing to use Benchopt instead of WandB to make it easier to run.

## Replicating the analyses / figures

All the R code used to generate the analyses and figures in available in the `analyses` folder.


# Benchmarking your own algorithm

## Downloading the datasets

The datasets used in the benchmark have been uploaded as OpenML
benchmarks, with the same transformations that are used in the paper.

```
import openml
#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key
SUITE_ID = 297 # Regression on numerical features
#SUITE_ID = 298 # Classification on numerical features
#SUITE_ID = 299 # Regression on numerical and categorical features
#SUITE_ID = 300 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
```

## Using our results

If you want to compare you own algorithms with the models used in 
this benchmark for a given number of random search iteration,
you can use the results from our random searches, which we share 
as a csv file at this address.

## Using our code

To benchmark your own algorithm using our code, you'll need:

- a model which uses the sklearn's API, i.e having fit and predict methods.
We recommend using [Skorch](https://skorch.readthedocs.io/en/stable/net.html) use sklearn's API with a Pytorch model.
- to add your model hyperparameters search space to `launch_config/model_config`.
- to add your model name in `launch_config/launch_benchmarks` and `utils/keyword_to_function_conversion.py`
- to run the benchmarks as explained in **Training the models**.

We're planning to release a version allowing to use Benchopt instead of WandB to make it easier to run.


