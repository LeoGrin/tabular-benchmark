import wandb
import sys
sys.path.append(".")
from configs.all_model_configs import total_config


def create_sweep(data_transform_config, model_name, regression, default, project, name,
                 dataset_size, categorical, datasets, remove_tranforms_from_model_config=False,
                 max_val_samples=None, max_test_samples=None):
    # Use the appropriate model config
    model_config = total_config[model_name]["regression" if regression else "classif"]["default" if default else "random"]
    print(model_config)
    if remove_tranforms_from_model_config:  # prevent conflicts with data_transform_config
        model_config = model_config.copy()
        for key in list(model_config):
            if key.startswith("transform__"):
                del model_config[key]

    if "max_train_samples" not in data_transform_config.keys():
        if dataset_size == "small":
            data_transform_config["max_train_samples"] = {"value": 1000}
        elif dataset_size == "medium":
            data_transform_config["max_train_samples"] = {"value": 10000}
        elif dataset_size == "large":
            data_transform_config["max_train_samples"] = {"value": 50000}
        else:
            assert type(dataset_size) == int
            data_transform_config["max_train_samples"] = {"value": dataset_size}

    if max_val_samples is not None:
        data_transform_config["max_val_samples"] = {"value": max_val_samples}
    if max_test_samples is not None:
        data_transform_config["max_test_samples"] = {"value": max_test_samples}

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

    if "use_gpu" not in model_config.keys():
        use_gpu = False
    else:
        use_gpu = model_config["use_gpu"]["value"]
    return sweep_id, use_gpu
