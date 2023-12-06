import wandb

from utils import create_sweep
import pandas as pd
import sys
sys.path.append(".")
from configs.wandb_config import wandb_id
from configs.all_model_configs import total_config
import argparse
import openml
import numpy as np



data_transform_config = {
    "data__method_name": {
        "value": "openml_no_transform"
    },
    "n_iter": {
        "value": "auto",
    },
}

benchmarks = [{"task": "regression",
                   "dataset_size": "medium",
                   "categorical": False,
                    "name": "numerical_regression",
                   "suite_id": 336,
                   "exclude": []},

                {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "name": "numerical_regression_large",
                    "suite_id": 336,
                    "exclude": []},

                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": False,
                    "name": "numerical_classification",
                    "suite_id": 337,
                    "exlude": []
                 },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": False,
                    "name": "numerical_classification_large",
                    "suite_id": 337,
                    "exclude": []
                 },

                {"task": "regression",
                    "dataset_size": "medium",
                    "categorical": True,
                    "name": "categorical_regression",
                    "suite_id": 335,
                    "exclude": [],
                },

                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                    "name": "categorical_regression_large",
                    "suite_id": 335,
                    "exclude": [],},

                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": True,
                    "name": "categorical_classification",
                    "suite_id": 334,
                    "exclude": [],
                 },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": True,
                    "name": "categorical_classification_large",
                    "suite_id": 334,
                    "exclude": [],
                 }
]

if __name__ == "__main__":
    # Create a csv file with all the WandB sweeps
    #Make an argparse
    parser = argparse.ArgumentParser()
    # List of benchmarks as argument
    parser.add_argument("--benchmarks", nargs="+", type=str, default=["numerical_classification", "numerical_regression", "categorical_classification", "categorical_regression"])
    # List of models as argument
    parser.add_argument("--models", nargs="+", type=str, default=[])
    # output file name
    parser.add_argument("--output_file", type=str, default="all_benchmark_medium.csv")
    # Datasets
    parser.add_argument("--datasets", nargs="+", type=str, default=[])
    # Exclude datasets
    parser.add_argument("--exclude", nargs="+", type=str, default=[])
    # Suffix for project name
    parser.add_argument("--suffix", type=str, default="")
    # Only default
    parser.add_argument("--default", action="store_true")


    args = parser.parse_args()

    if len(args.models) == 0:
        models = list(total_config.keys())
    else:
        models = args.models
    print(models)
    print(args.benchmarks)
    benchmark_names = args.benchmarks
    output_filename = args.output_file
    sweep_ids = []
    names = []
    projects = []
    use_gpu_list = []
    n_datasets_list = []
    benchmarks = [benchmark for benchmark in benchmarks if benchmark["name"] in benchmark_names]
    print(benchmarks)
    default_list = [False, True] if not args.default else [True]

    for n in range(1):
        for model_name in models:
            project_name = model_name + "_benchmark" + args.suffix 
            wandb.init(entity=wandb_id, project=project_name) # Create a new project on your WandB account
            for i, benchmark in enumerate(benchmarks):
                for default in default_list:
                    name = f"{model_name}_{benchmark['task']}_{benchmark['dataset_size']}"
                    if benchmark['categorical']:
                        name += "_categorical"
                    else:
                        name += "_numerical"
                    random_suffix = np.random.randint(10000)
                    name += f"_{random_suffix}"
                    if default:
                        name += "_default" # do not change
                    name += "_{}".format(n)
                    suite_id = benchmark["suite_id"]
                    # Get task_ids from openml suite
                    datasets = [int(dataset) for dataset in args.datasets]
                    exclude = [int(dataset) for dataset in args.exclude]
                    task_ids = openml.study.get_suite(suite_id).tasks
                    print("task_ids", task_ids)
                    print("datasets", datasets)
                    if len(args.datasets) == 0:
                        datasets_to_use = [id for id in task_ids if id not in exclude]
                    else:
                        datasets_to_use = [dataset for dataset in datasets if dataset in task_ids]
                        assert len(args.exclude) == 0
                    print("datasets", datasets_to_use)
                    sweep_id, use_gpu = create_sweep(data_transform_config,
                                 model_name=model_name,
                                 regression=benchmark["task"] == "regression",
                                 categorical=benchmark["categorical"],
                                 dataset_size = benchmark["dataset_size"],
                                 datasets = datasets_to_use,
                                 default=default,
                                 project=project_name,
                                 name=name)
                    n_datasets = len(datasets_to_use)
                    sweep_ids.append(sweep_id)
                    names.append(name)
                    projects.append(project_name)
                    use_gpu_list.append(use_gpu)
                    n_datasets_list.append(n_datasets)
                    print(f"Created sweep {name}")
                    print(f"Sweep id: {sweep_id}")
                    print(f"In project {project_name}")
                    print(f"Use GPU: {use_gpu}")
                    print(f"Num datasets: {n_datasets}")

    df = pd.DataFrame({"sweep_id": sweep_ids, "name": names, "project":projects, "use_gpu": use_gpu_list,
                       "n_datasets": n_datasets_list})
    df.to_csv(f"launch_benchmarks/sweeps/{output_filename}.csv", index=False)
    print("Check the sweeps id saved at launch_benchmarks/sweeps/{}.csv".format(output_filename))
    print("You can now run each sweep with wandb agent <USERNAME/PROJECTNAME/SWEEPID>, or use launch_on_cluster.py "
          "after making a few changes")


