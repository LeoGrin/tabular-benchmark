from launch_benchmarks.model_configs import config_dic
from run_experiment import train_model_on_config
import os
os.environ["WANDB_MODE"]="offline"


def test_models():
    # Run all models on one dataset per benchmark for a few epochs
    # to check that everything is working
    test_benchmarks = [
                       #  {"task": "regression",
                       #  "dataset_size": "medium",
                       #  "categorical": False,
                       #  "datasets": ["fifa"]},
                       #
                       # {"task": "regression",
                       #  "dataset_size": "large",
                       #  "categorical": False,
                       #  "datasets": ["house_16H"]},
                       #
                       #  {"task": "classif",
                       #  "dataset_size": "medium",
                       #  "categorical": False,
                       #  "datasets": ["phoneme"]
                       #  },
                       #
                       # {"task": "classif",
                       #  "dataset_size": "large",
                       #  "categorical": False,
                       #  "datasets": ["MiniBooNE"],
                       #  },

                       {"task": "regression",
                        "dataset_size": "medium",
                        "categorical": True,
                        "datasets": ["diamonds"]},

                       {"task": "regression",
                        "dataset_size": "large",
                        "categorical": True,
                        "datasets": ["black_friday"]},

                       {"task": "classif",
                        "dataset_size": "medium",
                        "categorical": True,
                        "datasets": ["electricity"]
                        },

                       {"task": "classif",
                        "dataset_size": "large",
                        "categorical": True,
                        "datasets": ["covertype"]
                        }
                           ]

    data_transform_config = {
        "data__method_name": {
            "value": "real_data"
        },
        "n_iter": {
            "value": "auto",
        },
    }

    models = ["gbt", "rf", "xgb", "hgbt",
              "ft_transformer", "resnet", "mlp", "saint"]

    for benchmark in test_benchmarks:
        for model_name in models:
            print(model_name)
            print("on")
            print(benchmark)
            # Use the appropriate model config
            model_config = config_dic[model_name]["regression" if benchmark["task"] == "regression" else "classif"][
                "default"]

            if benchmark["dataset_size"] == "medium":
                data_transform_config["max_train_samples"] = {"value": 10000}
            elif benchmark["dataset_size"] == "large":
                data_transform_config["max_train_samples"] = {"value": 50000}
            else:
                assert type(benchmark["dataset_size"]) == int
                data_transform_config["max_train_samples"] = {"value": benchmark["dataset_size"]}

            if benchmark["categorical"]:
                data_transform_config["data__categorical"] = {"value": True}
            else:
                data_transform_config["data__categorical"] = {"value": False}

            if benchmark["task"] == "regression":
                data_transform_config["regression"] = {"value": True}
                data_transform_config["data__regression"] = {"value": True}
            else:
                data_transform_config["regression"] = {"value": False}
                data_transform_config["data__regression"] = {"value": False}

            data_transform_config["data__keyword"] = {"values": benchmark["datasets"]}

            config_wandb = dict(model_config, **data_transform_config)
            # Translate from wandb config to config, i.e {"value": a} --> a
            config = {}
            for key in config_wandb.keys():
                new_value = list(config_wandb[key].values())[0]
                # to handle both {"values":[a]} and {"value":a}
                if type(new_value) == list:
                    new_value = new_value[0]
                config[key] = new_value
            print(config)

            # Make the training fast enough for a test
            # and make it run on cpu
            if config["model_type"] == "skorch":
                config["model__max_epochs"] = 1
                config["model__device"] = "cpu"
            elif config["model_type"] == "tab_survey":
                config["model__args__epochs"] = 1
                config["model__args__use_gpu"] = False
            config["n_iter"] = 1

            if train_model_on_config(config) != 0:
                raise ValueError('An error happened')


if __name__ == "__main__":
    test_models()