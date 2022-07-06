import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "rf_numeric_regression_large",
  "project": "thesis-4",
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
      "values": ["diamonds",
                  "nyc-taxi-green-dec-2016",
                 "year"]
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
    "transformed_target": {
        "values": [False, True]
    },
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-4")