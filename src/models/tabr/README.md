# TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning<!-- omit in toc -->

This is the official implementation of the paper "TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning" ([arXiv](https://arxiv.org/abs/2307.14338)).

Table of Contents:
- [The main results](#the-main-results)
- [How to reproduce the results](#how-to-reproduce-the-results)
    - [Set up the environment](#set-up-the-environment)
        - [Software](#software)
        - [Data](#data)
        - [Environment variables](#environment-variables)
    - [Quick test](#quick-test)
    - [Tutorial](#tutorial)
    - [Reproducing other results](#reproducing-other-results)
- [Understanding the repository](#understanding-the-repository)
    - [Code overview](#code-overview)
    - [Running scripts](#running-scripts)
    - [Technical notes](#technical-notes)
- [Adding new datasets and metrics](#adding-new-datasets-and-metrics)
    - [How to add a new dataset](#how-to-add-a-new-dataset)
    - [How to optimize a custom metric](#how-to-optimize-a-custom-metric)
    - [How to add a new task type](#how-to-add-a-new-task-type)

# The main results

After setting up the environment, use [this notebook](notebooks/results.ipynb) to browse the main results (for now, you can scroll to the last cell to get an idea of what it looks like).

# How to reproduce the results

## Set up the environment

### Software

For this project, we highly recommend using a conda-like environment manager instead of pip to get things right for the libraries that use CUDA, especially for Faiss.
The available options:
- [mamba](https://mamba.readthedocs.io/en/latest/installation.html) is a fast replacement for conda
- (we used this) [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#manual-installation) can be used to avoid any conflicts with your current setup: it is a single binary which does not require any "installation" (see the [documentation](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html))
- [conda](https://docs.conda.io/en/latest/miniconda.html) is a valid option, but setting up the environment can become extremely slow (or even impossible)

Then, run the following commands (replace `micromamba` with `mamba` or `conda` if needed):
```shell
git clone https://github.com/yandex-research/tabular-dl-tabr
cd tabular-dl-tabr
micromamba create -f environment.yaml
micromamba activate tabr
```

If the `micromamba create` command fails, try using `environment-simple.yaml` instead of `environment.yaml`.
If your machine does not have GPUs, use `environment-simple.yaml`, but replace `faiss-gpu` with `faiss-cpu` and remove `pytorch-cuda`.

### Data

(***License:** we do not impose any new license restrictions in addition to the original licenses of the used dataset.
See the paper to learn about the dataset sources*)

Navigate to the repository root and run the following commands:
```
wget https://huggingface.co/datasets/puhsu/tabular-benchmarks/resolve/main/data.tar -O tabular-dl-tabr.tar.gz
tar -xvf tabular-dl-tabr.tar.gz
```

After that, the `data/` directory should appear.

### Environment variables

**When running scripts, the environment variable `CUDA_VISIBLE_DEVICES` must be explicitly set**. So we assume that you do run the following command first before running other commands:

```
export CUDA_VISIBLE_DEVICES="0"
```

## Quick test

To check that the environment is configured correctly, run the following command and wait for the training to finish (in this experiment, hyperparameters and results are extremely suboptimal, this is needed only to test the environment):
```
python bin/ffn.py exp/debug/0.toml --force
```

The last line of the output log should look like this:
```
[<<<] exp/debug/0 | <date & time>
```

## Tutorial

Here, we reproduce the results for MLP on the California Housing dataset (in the paper, this dataset is referred to as "CA").
*Reproducing the results for other algorithms and datasets is very similar with rare exceptions, which are commented in further sections.*

The detailed description of the repository is provided later in the ["Understanding the repository"](#understanding-the-repository) section.
Until then, simply copying and pasting the instructions should just work.

Technically, reproducing the results for MLP on the California Housing dataset means reproducing the content of these directories:
1. `exp/mlp/california/0-tuning` is the result of the hyperparameter tuning
2. `exp/mlp/california/0-evaluation` is the result of evaluation of the tuned configuration from the previous step. This configuration is evaluated under 15 random seeds, which produces 15 single models.
3. `exp/mlp/california/0-ensemble-5` is the result of ensembles of the single models from the previous step (three disjoint ensembles each consisting of five models).

To reproduce the above results, run the following commands (takes up to 30-60 minutes on a single GPU):

```
cp exp/mlp/california/0-tuning.toml exp/mlp/california/0-reproduce-tuning.toml
python bin/go.py exp/mlp/california/0-reproduce-tuning.toml
```

In fact, `0-reproduce-tuning` is an arbitrary name and you can choose a different one, but it must end with `-tuning`.
Once the run is finished, the following directories should appear:
- `exp/mlp/california/0-reproduce-tuning`
- `exp/mlp/california/0-reproduce-evaluation`
- `exp/mlp/california/0-reproduce-ensemble-5`

After that, you can go to `notebooks/results.ipynb` and view your results (see the instructions just before the last cell of that notebook).

Note that `bin/go.py` is just a shortcut and the above commands are equivalent to this:

```
cp exp/mlp/california/0-tuning.toml exp/mlp/california/0-reproduce-tuning.toml
python bin/tune.py exp/mlp/california/0-reproduce-tuning.toml
python bin/evaluate.py exp/mlp/california/0-reproduce-tuning --function bin.ffn.main
python bin/ensemble.py exp/mlp/california/0-reproduce-evaluation
```

## Reproducing other results

**General comments**:
- To reiterate, for most models, the pipeline for reproducing the results is the same as for MLP in the above tutorial. Here, we only cover exceptions from this pattern.
- The last cell of `notebooks/results.ipynb` covers many (but not all) results from the paper with their locations in `exp/`.

**Evaluating specific configurations without tuning**.
To evaluate a specific set of hyperparameters without tuning, you can use `bin/go.py` (to evaluate single models and ensembles) or `bin/evaluate.py` (to evaluate only single models).
For example, this is how you can reproduce the results for the default XGBoost on the California Housing dataset:

```
mkdir exp/xgboost_/california/default2-reproduce-evaluation
cp exp/xgboost_/california/default2-evaluation/0.toml exp/xgboost_/california/default2-reproduce-evaluation/0.toml
python bin/go.py exp/xgboost_/california/default2-reproduce-evaluation --function bin.xgboost_.main
```

Note that now we have to explicitly pass the function that is being evaluated (`--function bin.xgboost_.main`).
Again, `default2-reproduce-evaluation` is an arbitrary name, the only requirement is that it ends with `-evaluation`.

**Custom versions of TabR**.
In `bin/`, there are several versions of the model.
Each of them has a corresponding directory in `exp/` with configs and results.
See ["Code overview"](#code-overview) to learn more.

**k Nearest Neighbors**.
To reproduce the results on the California Housing dataset:

```
cp exp/neighbors/california/0.toml exp/neighbors/california/0-reproduce.toml
python bin/neighbors.py exp/neighbors/california/0-reproduce.toml

mkdir exp/knn/california/0-reproduce-evaluation
cp exp/knn/california/0-evaluation/0.toml exp/knn/california/0-reproduce-evaluation/0.toml
python -c "
path = 'exp/knn/california/0-reproduce-evaluation/0.toml'
with open(path) as f:
    config = f.read()
with open(path, 'w') as f:
    f.write(config.replace(
        ':exp/neighbors/california/0',
        ':exp/neighbors/california/0-reproduce'
    ))
"
python bin/knn.py exp/knn/california/0-reproduce-evaluation/0.toml
```

**DNNR**.
First, you need to run `bin/dnnr_precompute_scaling.py` and obtain results similar to `exp/dnnr/precomputed_scaling` ("loo" and "ohe" differ only in how the categorical features are encoded; we choose the best of the two approaches on the next step based on the performance on the validation set).
Then, you need to run `bin/dnnr.py`, the corresponding configs are located in `exp/dnnr/<dataset name>`

**NPT**.
To evaluate NPT, we use [the official repository](https://github.com/OATML/non-parametric-transformers) with modifications to allow using our datasets and preprocessing.

# Understanding the repository

Read this if you are going to do more experiments/research in this repository.

## Code overview
- `bin` contains high-level scripts which produce the main results
    - Models
        - `tabr.py` is the "main" implemention of TabR with many useful technical comments inside
        - `tabr_scaling.py` is the version of `tabr.py` with the support for the "context freeze" technique described in the paper
        - `tabr_design.py` is the version of `tabr.py` with more options for testing various design decisions and doing ablation studies
        - `tabr_add_candidates_after_training.py` is the version of `tabr.py` for evaluating the addition of new unseen candidates after the training as described in the paper
        - `ffn.py` implements the general "feed-forward network" approach (currently, only the MLP backbone is available, but adding new backbones is simple)
        - `ft_transformer.py` implements FT-Transformer from the "Revisiting Deep Learning Models for Tabular Data" paper
        - `xgboost_.py` implements XGBoost
        - `lightgbm_.py` implements LightGBM
        - `catboost_.py` implements CatBoost
        - `neighbors.py` + `knn.py` implement k Nearest Neighbors
        - `dnnr_precompute_scaling.py` + `dnnr.py` implement DNNR from the "DNNR: Differential Nearest Neighbors Regression" paper
        - `saint.py` implements SAINT from the "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training" paper
        - `anp.py` implements the model from the "Attentive Neural Processes" paper
        - `dkl.py` implements the model from the "Deep Kernel Learning" paper
    - Infrastructure
        - `tune.py` tunes hyperparameters
        - `evaluate.py` evaluates a given config over multiple (by default, 15) random seeds
        - `ensemble.py` ensembles predictions produced by `evaluate.py`
        - `go.py` is a shorcut combining `[tune.py + evaluate.py + ensemble.py]`
- `notebooks` contains Jupyter notebooks
- `lib` contains common tools used by the scripts in `bin` and the notebooks in `notebooks`
- `exp` contains experiment configs and results (metrics, tuned configurations, etc.)
    - usually, for a given script in `bin`, there is a corresponding directory in `env`. However, this is just a convention, and you can have any layout in `exp`.

## Running scripts

For most scripts in `bin`, the pattern is as follows:

```
python bin/some_script.py exp/a/b/c.toml
```

When the run is successfully finished, the result will be the `exp/a/b/c` folder.
In particular, the `exp/a/b/c/DONE` file will be created.
Usually, the main part of the result is the `exp/a/b/c/report.json` file.

If you want to run the script with the same config again and **overwrite the existing results**, use the `--force` flag:

```
python bin/some_script.py exp/a/b/c.toml --force
```

Some scripts (`bin/tune.py` and `bin/go.py`) support the `--continue` flag.

The following scripts have command line interface instead of configs:
- `bin/go.py`
- `bin/evaluate.py`
- `bin/ensemble.py`

## Technical notes
- **(IMPORTANT)** For most algorithms, the configs are expected to have the `data` section which describes the input dataset
    - For regression problems, always set `y_policy = "standard"` unless you are absolutely sure that you need other value
    - Unless a given deep learning algorithm is special in some way, for a given dataset, the `data` section should be copied from the MLP config for the same dataset. For example, for California Housing dataset, this "source of truth" for deep learning algorithms is the `exp/mlp/california/0-tuning.toml` config.
- **(IMPORTANT)** For deep learning algorithms, for each dataset, the batch size is predefined. As in the previous bullet, the configs for MLP is the source of truth.
- For saving and loading configs programatically, use the `lib.dump_config` and `lib.load_config` functions (defined in `lib/util.py`) instead of bare TOML libraries.
- In many configs, you can see that path-like values (e.g. a path to a dataset) start with ":". It means "relative to the repository root", and this is handled by the `lib.get_path` function (defined in `lib/env.py`).
- The scripts in `bin` can be used as modules if needed: `import bin.ffn`. For example, this is used by `bin/evaluate.py` and `bin/tune.py`.

# Adding new datasets and metrics

## How to add a new dataset

To apply the scripts from this repository to your custom dataset, you need to create a new directory in the `data/` directory and **use the same file names and data types** as in our datasets.
A good example is the `data/adult` dataset where all supported feature types are presented (numerical, binary and categorical).
The `.npy` files are NumPy arrays saved with the `np.save` function ([documentation](https://numpy.org/doc/stable/reference/generated/numpy.save.html)).

Let's say your dataset is called `my-dataset`.
Then, create the `data/my-dataset` directory with the following content:
- If the dataset has numerical (i.e. continuous) features
    - Files: `X_num_train.npy`, `X_num_val.npy`, `X_num_test.npy`
    - NumPy data type: `np.float32`
- If the dataset has binary features
    - Files: `X_bin_train.npy`, `X_bin_val.npy`, `X_bin_test.npy`
    - NumPy data type: `np.float32`
    - All values must be `0.0` and `1.0`
- If the dataset has categorical features
    - Files: `X_cat_train.npy`, `X_cat_val.npy`, `X_cat_test.npy`
    - NumPy data type: `np.str_` (**yes, the values must be strings**)
- Labels
    - Files: `Y_train.npy`, `Y_val.npy`, `Y_test.npy`
    - NumPy data type: `np.float32` for regression, `np.int64` for classification
    - For classification problems, the labels must form the range `[0, ..., n_classes - 1]`.
- `info.json` -- a JSON file with the following keys:
    - `"task_type"`: one of `"regression"`, `"binclass"`, `"multiclass"`
    - (optional) `"name"`: any string (a "pretty" name for your dataset, e.g. `"My Dataset"`)
    - (optional) `"id"`: any string (must be unique among all `"id"` keys of all `info.json` files of all datasets in `data/`)
- `READY` -- just an empty file

At this point, your dataset is ready to use!

## How to optimize a custom metric
The "main" metric which is optimized in this repository is referred to as "score".
**Score is always maximized.**
By default:
- for regression problems, the score is negative RMSE
- for classification problems, the score is accuracy

In the `_SCORE_SHOULD_BE_MAXIMIZED` dictionary in `lib/data.py`, you can find other supported scores.
To use any of them, set the "score" field in the `[data]` section of a config:
```
...

[data]
seed = 0
path = ":data/california"
...
score = "r2"

...
```

To implement a custom metric, add its name to the `_SCORE_SHOULD_BE_MAXIMIZED` dictionary and compute it in the `lib/metrics.py:calculate_metrics` function.

## How to add a new task type

We do not provide instructions for that.
While adding new task types is definitely possible, overall, the code is written without other task types in mind.
For example, there may be places where the code implicitly assumes that the task is either regression or classification.
So adding a new task type will require carefully reviewing the whole codebases to find places where the new task type should be taken into account.

# How to cite<!-- omit in toc -->

```
@article{gorishniy2023tabr,
    title={TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning},
    author={
        Yury Gorishniy and
        Ivan Rubachev and
        Nikolay Kartashev and
        Daniil Shlenskii and
        Akim Kotelnikov and
        Artem Babenko
    },
    journal={arXiv},
    volume={2307.14338},
    year={2023},
}
```
