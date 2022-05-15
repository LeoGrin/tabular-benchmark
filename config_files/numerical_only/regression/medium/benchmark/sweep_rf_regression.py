import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "rf_benchmark_numeric_regression",
  "project": "thesis",
  "method" : "random",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "rf_r"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__n_estimators": {
      "distribution": "q_log_uniform_values",
      "min": 9.5,
      "max": 3000.5,
      "q": 1
    },
    "model__criterion": {
      "values": ["mse"],
    },
    "model__max_features": { # like Hyperopt ?
      "values": ["sqrt", "sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    "model__max_depth": { # Not exactly like Hyperopt
      "values": [None, 2, 3, 4],
    },
    "model__min_samples_split": {
      "value": 2
    },
    "model__min_samples_leaf": { # Not exactly like Hyperopt
      "distribution": "q_log_uniform_values",
      "min": 1.5,
      "max": 50.5,
      "q": 1
    },
    "model__bootstrap": {
      "values": [True, False]
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["cpu_act",
                 "pol",
                 "elevators",
                 "isolet",
                 "wine_quality",
                  "Ailerons",
                  "yprop_4_1",
                  "houses",
                  "house_16H",
                  #"delays_zurich_transport",
                  #"diamonds",
                  #"Allstate_Claims_Severity",
                  "Bike_Sharing_Demand",
                  "OnlineNewsPopularity",
                  #"nyc-taxi-green-dec-2016",
                  "house_sales",
                  "sulfur",
                  "Bike_Sharing_Demand",
                  #"fps-in-video-games",
                  #"medical_charges",
                  "MiamiHousing2016",
                  "superconduct"]
    },
    "n_iter": {
      "value": "auto",
    },
    "regression": {
      "value": True
    },
    "data__regression": {
      "value": True
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")