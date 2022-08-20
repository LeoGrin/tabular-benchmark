from launch_benchmarks import benchmarks, data_transform_config
import wandb
from model_configs import config_dic
import numpy as np
import pandas as pd

WANDB_PROJECT_NAMES = ["bo_1", "bo_2", "bo_3", "bo_4"]

def convert_proba_to_repeated_list(values, probabilities):
    # Due to a weird bug in wandb, we need to convert the values, probabilities tuple to a list with repeated elements
    min_proba = min(probabilities)
    multiplier = 10 ** (np.ceil(- np.log10(min_proba)))
    print(values, probabilities, multiplier)
    return sum([[values[i]] * int(probabilities[i] * multiplier) for i in range(len(values))], [])

def create_sweep_bo(data_transform_config, model_name, regression, project, name,
                 dataset_size, categorical, dataset, remove_tranforms_from_model_config=False):
    # Use the appropriate model config
    model_config = config_dic[model_name]["regression" if regression else "classif"]["random"]
    if remove_tranforms_from_model_config:  # prevent conflicts with data_transform_config
        model_config = model_config.copy()
        for key in model_config.keys():
            if key.startswith("transform__"):
                del model_config[key]

    if dataset_size == "medium":
        data_transform_config["max_train_samples"] = {"value": 10000}
    elif dataset_size == "large":
        data_transform_config["max_train_samples"] = {"value": 50000}
    else:
        assert type(dataset_size) == int
        data_transform_config["max_train_samples"] = {"value": dataset_size}

    if categorical:
        data_transform_config["data__categorical"] = {"value": True}
    else:
        data_transform_config["data__categorical"] = {"value": False}

    if regression:
        data_transform_config["regression"] = {"value": True}
        data_transform_config["data__regression"] = {"value": True}
    else:
        data_transform_config["regression"] = {"value": False}
        data_transform_config["data__regression"] = {"value": False}

    data_transform_config["data__keyword"] = {"value": dataset}

    for key, values in model_config.items():
        if "probabilities" in values.keys():
            model_config[key]["values"] = convert_proba_to_repeated_list(values["values"], values["probabilities"])
            del model_config[key]["probabilities"]

    sweep_config = {
        "program": "run_experiment.py",
        "name": name,
        "project": project,
        "method": "bayes",
        "metric": {
            "name": "mean_test_score",
            "goal": "minimize"  # RMSE
        } if regression else {
            "name": "mean_test_score",
            "goal": "maximize"  # accuracy
        },
        "parameters": dict(model_config, **data_transform_config)
    }

    sweep_id = wandb.sweep(sweep_config, project=project)

    return sweep_id



if __name__ == "__main__":
    models = ["gbt", "xgb", "hgbt"]  # ,
    # "ft_transformer", "resnet", "mlp", "saint"]
    benchmarks_medium = [benchmark for benchmark in benchmarks if benchmark["dataset_size"] == "medium"]
    sweep_ids = []
    names = []
    projects = []
    for i, benchmark in enumerate(benchmarks_medium):
        for model_name in models:
            for dataset in benchmark["datasets"]:
                name = f"{model_name}_{benchmark['task']}_{benchmark['dataset_size']}_{dataset}"
                if benchmark['categorical']:
                    name += "_categorical"
                else:
                    name += "_numerical"
                sweep_id = create_sweep_bo(data_transform_config,
                                           model_name=model_name,
                                           regression=benchmark["task"] == "regression",
                                           categorical=benchmark["categorical"],
                                           dataset_size=benchmark["dataset_size"],
                                           dataset=dataset,
                                           project=WANDB_PROJECT_NAMES[i],
                                           name=name)
                print(f"Created sweep {name}")
                print(f"Sweep id: {sweep_id}")
                sweep_ids.append(sweep_id)
                names.append(name)
                projects.append(WANDB_PROJECT_NAMES[i])
                print(f"Created sweep {name}")
                print(f"Sweep id: {sweep_id}")
                print(f"In project {WANDB_PROJECT_NAMES[i]}")

    df = pd.DataFrame({"sweep_id": sweep_ids, "name": names, "project":projects})
    file_name = "bo_trees"
    df.to_csv(f"launch_config/sweeps/{file_name}.csv", index=False)
    print(f"Check the sweeps id saved at sweeps/{file_name}.csv")
