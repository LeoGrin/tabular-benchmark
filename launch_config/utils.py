import wandb
from model_configs import config_dic


def create_sweep(data_transform_config, model_name, regression, default, project, name,
                 dataset_size, categorical, datasets, remove_tranforms_from_model_config=False):
    # Use the appropriate model config
    model_config = config_dic[model_name]["regression" if regression else "classif"]["default" if default else "random"]
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

    data_transform_config["data__keyword"] = {"values": datasets}

    sweep_config = {
        "program": "run_experiment.py",
        "name": name,
        "project": project,
        "method": "grid" if default else "random",
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
