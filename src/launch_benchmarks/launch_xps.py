from utils import create_sweep
import pandas as pd
import sys
sys.path.append(".")
from configs.wandb_config import wandb_id
import wandb

data_transform_config = {
    "data__method_name": {
        "value": "openml_no_transform"
    },
    "n_iter": {
        "value": "auto",
    },
}

xps = [{"name": "random_rotation",
       "config": {
            "transform__0__method_name": {
                "value": "remove_features_rf"
            },
            "transform__0__num_features_to_remove": {
                "values": [0.0, 0.5],
            },
            "transform__0__model_to_use": {
                "values": ["rf_c"],
            },
            "transform__1__method_name": {
                "value": "gaussienize"
            },
            "transform__1__type": {
                "value": "quantile",
            },
            "transform__2__method_name": {
                "value": "random_rotation"
            },
            "transform__2__deactivated": {
                "values": [True, False]
            },
        }
    },
    {"name": "smooth_target",
         "config": {
            "transform__0__method_name": {
                "value": "gaussienize",
            },
            "transform__0__type": {
                "value": "quantile",
            },
            "transform__1__method_name": {
                "value": "select_features_rf",
            },
            "transform__1__num_features": {
                "value": 5,
            },
            "transform__2__method_name": {
                "value": "remove_high_frequency_from_train",
            },
            "transform__2__cov_mult": {
                "values": [0.05]
            },
            "transform__2__covariance_estimation": {
                "values": ["robust"]
            },
         }
     },
    {"name": "add_features",
            "config": {
                "transform__0__method_name": {
                    "value": "add_uninformative_features"
                },
                "transform__0__multiplier": {
                    "values": [1., 1.5, 2],
                },
            }
     },
    {"name": "remove_features",
            "config": {
                "transform__0__method_name": {
                    "value": "remove_features_rf"
                },
                "transform__0__num_features_to_remove": {
                    "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                },
                "transform__0__model_to_use": {
                    "values": ["rf_c"],
                },
                "transform__0__keep_removed_features": {
                    "value": [True, False]
                },
            }
     }
]



models = ["gbt", "rf",
          "ft_transformer", "resnet"]

datasets = ["electricity",
             "covertype",
             "pol",
             "house_16H",
             "kdd_ipums_la_97-small",
             "MagicTelescope",
             "bank-marketing",
             "phoneme",
             "MiniBooNE",
             "Higgs",
             "eye_movements",
             "credit",
             "california",
             "wine"]

if __name__ == "__main__":
    sweep_ids = []
    names = []
    projects = []
    use_gpu_list = []
    n_datasets_list = []
    for i, xp in enumerate(xps):
        project_name =  xp["name"]
        wandb.init(project = project_name, entity=wandb_id)
        for model_name in models:
            for default in [True, False]:
                name = f"{model_name}_{xp['name']}"
                if default:
                    name += "_default"
                if model_name in ["ft_transformer", "resnet"]:
                    if xp["name"] in ["add_features", "remove_features"]:
                        data_transform_config["transform__1__method_name"] = {
                            "value": "gaussienize"
                        }
                        data_transform_config["transform__1__type"] = {
                            "value": "quantile"
                        }
                sweep_id, use_gpu = create_sweep(data_transform_config,
                             model_name=model_name,
                             regression=False,
                             categorical=False,
                             dataset_size = "medium",
                             datasets = datasets,
                             default=default,
                             project=project_name,
                             name=name,
                            remove_tranforms_from_model_config=True) #overwrite transforms in model config
                sweep_ids.append(sweep_id)
                names.append(name)
                projects.append(project_name)
                use_gpu_list.append(use_gpu)
                n_datasets_list.append(len(datasets))
                print(f"Created sweep {name}")
                print(f"Sweep id: {sweep_id}")
                print(f"Project: {project_name}")
                print(f"Use GPU: {use_gpu}")
                print(f"Number of datasets: {len(datasets)}")

    df = pd.DataFrame({"sweep_id": sweep_ids, "name": names,
                       "project": projects,
                       "use_gpu": use_gpu_list,
                       "n_datasets": n_datasets_list})
    df.to_csv("launch_benchmarks/sweeps/xps_sweeps.csv", index=False)
    print("Check the sweeps id saved at sweeps/xps_sweeps.csv")
    print("You can now run each sweep with wandb agent <USERNAME/PROJECTNAME/SWEEPID>, or use launch_on_cluster.py "
          "after making a few changes")

