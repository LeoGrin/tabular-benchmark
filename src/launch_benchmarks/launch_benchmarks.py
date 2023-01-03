import wandb

from utils import create_sweep
import pandas as pd
import sys
sys.path.append(".")
from configs.wandb_config import wandb_id
from configs.all_model_configs import total_config
import argparse
import openml 



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
                   "suite_id": None,
                   "exclude": []},

                {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "name": "numerical_regression_large",
                    "datasets": ["diamonds",
                                  "nyc-taxi-green-dec-2016",
                                 "year"]},

                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": False,
                    "name": "numerical_classification",
                    "suite_id": 329,
                    "exlude": []
                 },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": False,
                    "name": "numerical_classification_large",
                    "datasets": ["covertype",
                                 "MiniBooNE",
                                 "Higgs",
                                 "jannis"],
                 },

                {"task": "regression",
                    "dataset_size": "medium",
                    "categorical": True,
                    "name": "categorical_regression",
                 "datasets": ["yprop_4_1",
                             "analcatdata_supreme",
                             "visualizing_soil",
                             "black_friday",
                             "nyc-taxi-green-dec-2016",
                             "diamonds",
                             "Mercedes_Benz_Greener_Manufacturing",
                             "Brazilian_houses",
                             "Bike_Sharing_Demand",
                             "OnlineNewsPopularity",
                             "house_sales",
                             "particulate-matter-ukair-2017",
                             "SGEMM_GPU_kernel_performance"]},

                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                    "name": "categorical_regression_large",
                 "datasets": ["black_friday",
                     "nyc-taxi-green-dec-2016",
                     "diamonds",
                     "particulate-matter-ukair-2017",
                     "SGEMM_GPU_kernel_performance"]},

                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": True,
                    "name": "categorical_classification",
                    "datasets": ["electricity",
                                 "eye_movements",
                                  "KDDCup09_upselling",
                                  "covertype",
                                  "rl",
                                  "road-safety",
                                  "compass"]
                 },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": True,
                    "name": "categorical_classification_large",
                    "datasets": ["covertype",
                                 "road-safety"]
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

    for n in range(1):
        for model_name in models:
            project_name = model_name + "_benchmark" + args.suffix 
            wandb.init(entity=wandb_id, project=project_name) # Create a new project on your WandB account
            for i, benchmark in enumerate(benchmarks):
                for default in [False, True]:
                    name = f"{model_name}_{benchmark['task']}_{benchmark['dataset_size']}"
                    if benchmark['categorical']:
                        name += "_categorical"
                    else:
                        name += "_numerical"
                    if default:
                        name += "_default" # do not change
                    name += "_{}".format(n)
                    suite_id = benchmark["suite_id"]
                    # Get task_ids from openml suite
                    if len(args.datasets) == 0:
                        task_ids = openml.study.get_suite(suite_id).tasks
                        datasets_to_use = [id for id in task_ids if id not in args.exclude]
                    else:
                        datasets_to_use = args.datasets
                        assert len(args.exclude) == 0
                    sweep_id, use_gpu = create_sweep(data_transform_config,
                                 model_name=model_name,
                                 regression=benchmark["task"] == "regression",
                                 categorical=benchmark["categorical"],
                                 dataset_size = benchmark["dataset_size"],
                                 datasets = task_ids,
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


