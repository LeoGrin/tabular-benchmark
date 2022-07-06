import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "gpt_benchmark_numeric_classif_2",
  "project": "thesis-3",
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
      "value": "gbt_c"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__loss": {
      "values": ["deviance", "exponential"],
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
      "values": ["electricity",
                 "covertype",
                 #"poker",
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
    "n_iter": {
      "value": "auto",
    },
    "regression": {
      "value": False
    },
    "data__regression": {
      "value": False
    },
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-3")