import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "gpt_benchmark_regression_categorical",
  "project": "thesis-4",
  "method" : "random",
  "metric": {
    "name": "mean_test_score",
    "goal": "maximize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "gbt_r"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__loss": {
      "values": ["squared_error", "absolute_error", "huber"],
    },
    "model__alpha": {
      "distribution": "uniform",
        'min': 0.85,
        'max': 0.95,
    },
    "model__learning_rate": {
      'distribution': "log_normal",
      'mu': float(np.log(0.01)),
      'sigma': float(np.log(10.0)),
    },
    "model__subsample": { # Not exactly like Hyperopt-sklearn
      'distribution': "uniform",
      'min': 0.5,
      'max': 1.0,
    },
    "model__n_estimators": {
      "distribution": "q_log_uniform_values",
      "min": 10.5,
      "max": 1000.5,
      "q": 1
    },
    "model__criterion": {
      "values": ["friedman_mse", "squared_error"]
    },
    "model__max_depth": { # Not exactly like Hyperopt
      "values": [None, 2, 3, 4, 5],
      "probabilities": [0.1, 0.1, 0.6, 0.1, 0.1]
    },
    "model__min_samples_split": {
      "values": [2, 3],
      "probabilities": [0.95, 0.05]
    },
    "model__min_samples_leaf": { # Not exactly like Hyperopt
      "distribution": "q_log_uniform_values",
      "min": 1.5,
      "max": 50.5,
      "q": 1
    },
    "model__min_impurity_decrease": {
      "values": [0.0, 0.01, 0.02, 0.05],
      "probabilities": [0.85, 0.05, 0.05, 0.05],
    },
    "model__max_leaf_nodes": {
      "values": [None, 5, 10, 15],
      "probabilities": [0.85, 0.05, 0.05, 0.05]
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
    "one_hot_encoder": {
        "value": True
    },
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-4")