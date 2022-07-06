import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "hgbt_categorical_regression",
  "project": "thesis-3",
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
      "value": "hgbt_r"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__loss": {
      "values": ["squared_error", "absolute_error"],
    },
    "model__learning_rate": {
      'distribution': "log_normal",
      'mu': float(np.log(0.01)),
      'sigma': float(np.log(10.0)),
    },
    "model__max_leaf_nodes": {
        'distribution': "q_normal",
        'mu': 31,
        "sigma": 5
    },
    "model__max_depth": { # Added None compared to hyperopt
      "values": [None, 2, 3, 4],
      "probabilities": [0.1, 0.1, 0.7, 0.1]
    },
    "model__min_samples_leaf": { # Not exactly like Hyperopt
      "distribution": "q_normal",
      "mu": 20,
      "sigma": 2,
      "q": 1
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["yprop_4_1",
                  "analcatdata_supreme",
                  "visualizing_soil",
                  "black_friday",
                  "nyc-taxi-green-dec-2016",
                  "diamonds",
                  "Allstate_Claims_Severity",
                  "Mercedes_Benz_Greener_Manufacturing",
                  "Brazilian_houses",
                  "Bike_Sharing_Demand",
                  "OnlineNewsPopularity",
                  "house_sales",
                  "LoanDefaultPrediction",
                  "particulate-matter-ukair-2017",
                  "SGEMM_GPU_kernel_performance"]
    },
    "n_iter": {
      "value": "auto",
    },
    "regression": {
          "value": True
    },
    "data__regression": {
          "value": True
    },
    "data__categorical": {
      "value": True
    },
    "transformed_target": {
      "values": [False, True]
    },
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-3")