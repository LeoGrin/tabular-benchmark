from utils import create_sweep
import pandas as pd

# We use one project per benchmark to avoid WandB getting super slow
WANDB_PROJECT_NAMES = ["xgb_new"] * 10


data_transform_config = {
    "data__method_name": {
        "value": "real_data"
    },
    "n_iter": {
        "value": "auto",
    },
}

benchmarks = [{"task": "regression",
                   "dataset_size": "medium",
                   "categorical": False,
                   "datasets":  ["cpu_act",
                     "pol",
                     "elevators",
                     #"isolet",
                     "wine_quality",
                      "Ailerons",
                      "houses",
                      "house_16H",
                      "diamonds",
                      "Brazilian_houses",
                      "Bike_Sharing_Demand",
                      "nyc-taxi-green-dec-2016",
                      "house_sales",
                      "sulfur",
                      "medical_charges",
                      "MiamiHousing2016",
                      "superconduct",
                      "california",
                      "year",
                      "fifa"]},

                {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["diamonds",
                                  "nyc-taxi-green-dec-2016",
                                 "year"]},

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
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["covertype",
                                 "MiniBooNE",
                                 "Higgs",
                                 "jannis"],
                 },

                {"task": "regression",
                    "dataset_size": "medium",
                    "categorical": True,
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
                 "datasets": ["black_friday",
                     "nyc-taxi-green-dec-2016",
                     "diamonds",
                     "particulate-matter-ukair-2017",
                     "SGEMM_GPU_kernel_performance"]},

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
                 },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": True,
                    "datasets": ["covertype",
                                 "road-safety"]
                 }
]

#models = ["saint"]#["gbt", "rf", "xgb", "hgbt"]#,
          #"ft_transformer", "resnet", "mlp", "saint"]


if __name__ == "__main__":
    models = ["xgb", "gbt", "hgbt"]
    sweep_ids = []
    names = []
    projects = []
    #benchmarks_bonus = [benchmark for benchmark in benchmarks if benchmark["task"] == "regression"]
    #benchmarks_medium = [benchmark for benchmark in benchmarks_bonus if benchmark["dataset_size"] == "medium"]
    #datasets = ["wine_quality", "year", "cpu_act", "Bike_Sharing_Demand", "pol"]
    benchmarks = [benchmark for benchmark in benchmarks]

    for n in range(1):
        for model_name in models:
            for i, benchmark in enumerate(benchmarks):
                for default in [True]:
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
                    print(f"Created sweep {name}")
                    print(f"Sweep id: {sweep_id}")
                    print(f"In project {WANDB_PROJECT_NAMES[i]}")

    df = pd.DataFrame({"sweep_id": sweep_ids, "name": names, "project":projects})
    df.to_csv("launch_config/sweeps/boosting_10K_default.csv", index=False)
    print("Check the sweeps id saved at sweeps/benchmark_sweeps.csv")


