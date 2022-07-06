import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "useless_features_gbt_bonus",
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
                 #"jannis",
                 "credit",
                 "california",
                 "wine"]
    },
    "transform__0__method_name": {
      "value": "remove_features_rf"
    },
    "transform__0__num_features_to_remove": {
      "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "transform__0__model_to_use": {
      "values": ["rf_c"],
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