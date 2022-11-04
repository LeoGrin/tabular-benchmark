from utils import create_sweep
import pandas as pd

# We use one project per benchmark to avoid WandB getting super slow
WANDB_PROJECT_NAMES = ["tabpfn2"] * 10


data_transform_config = {
    "data__method_name": {
        "value": "real_data"
    },
    #"data__impute_nans": {
    #    "value": True
    #},
    #"data__balance":{
    #    "value": True
    #},
    "max_train_samples": {
        "values": [300, 1000]
    },
    "n_iter": {
        "value": "auto",
    },
}

benchmarks = [
#{"task": "classif",
#                     "dataset_size": "small",
#                     "categorical": True,
#                     "regression": False,
#                     "datasets": [11,
#                                                  14,
#                                                  15,
#                                                  16,
#                                                  18,
#                                                  22,
#                                                  23,
#                                                  29,
#                                                  31,
#                                                  37,
#                                                  50,
#                                                  54,
#                                                  188,
#                                                  458,
#                                                  469,
#                                                  1049,
#                                                  1050,
#                                                  1063,
#                                                  1068,
#                                                  1510,
#                                                  1494,
#                                                  1480,
#                                                  1462,
#                                                  1464,
#                                                  6332,
#                                                  23381,
#                                                  40966,
#                                                  40982,
#                                                  40994,
#                                                  40975]}]

{"task": "classif",
 "dataset_size": "medium",
 "categorical": False,
 "datasets": ["electricity",
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
              "jannis",
              "credit",
              "california",
              "wine"]
 },

{"task": "classif",
 "dataset_size": "medium",
 "categorical": True,
 "datasets": ["electricity",
              "eye_movements",
              "KDDCup09_upselling",
              "covertype",
              "rl",
              "road-safety",
              "compass"]
 }
]

#models = ["saint"]#["gbt", "rf", "xgb", "hgbt"]#,
          #"ft_transformer", "resnet", "mlp", "saint"]


if __name__ == "__main__":
    models = ["stacking"]
    sweep_ids = []
    names = []
    projects = []
    n_datasets = []
    #benchmarks_bonus = [benchmark for benchmark in benchmarks if benchmark["task"] == "regression"]
    #benchmarks_medium = [benchmark for benchmark in benchmarks_bonus if benchmark["dataset_size"] == "medium"]
    #datasets = ["wine_quality", "year", "cpu_act", "Bike_Sharing_Demand", "pol"]
    benchmarks = [benchmark for benchmark in benchmarks]

    for n in range(1):
        for model_name in models:
            for i, benchmark in enumerate(benchmarks):
                for default in [True]: #TODO
                    #if benchmark["task"] == "classif" and not benchmark["categorical"]:
                    #    continue
                    name = f"{model_name}_{benchmark['task']}_{benchmark['dataset_size']}"
                    if benchmark['categorical']:
                        name += "_categorical"
                    else:
                        name += "_numerical"
                    if default:
                        name += "_default"
                    name += "_{}".format(n)
                    sweep_id = create_sweep(data_transform_config,
                                 model_name=model_name,
                                 regression=benchmark["task"] == "regression",
                                 categorical=benchmark["categorical"],
                                 dataset_size = benchmark["dataset_size"],
                                 datasets = benchmark["datasets"],
                                 default=default,
                                 project=WANDB_PROJECT_NAMES[i],
                                 name=name)
                    sweep_ids.append(sweep_id)
                    names.append(name)
                    projects.append(WANDB_PROJECT_NAMES[i])
                    n_datasets.append(len(benchmark["datasets"]))
                    print(f"Created sweep {name}")
                    print(f"Sweep id: {sweep_id}")
                    print(f"In project {WANDB_PROJECT_NAMES[i]}")

    df = pd.DataFrame({"sweep_id": sweep_ids, "name": names, "project":projects,
                       "n_datasets": n_datasets})
    df.to_csv("launch_config/sweeps/stacking_tabpfn.csv", index=False)
    print("Check the sweeps id saved at sweeps/benchmark_sweeps.csv")


