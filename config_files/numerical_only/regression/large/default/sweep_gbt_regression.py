import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "gbt_numeric_regression_large",
  "project": "thesis-5",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "gbt_r"
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
        "values": [False]
    },
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-5")