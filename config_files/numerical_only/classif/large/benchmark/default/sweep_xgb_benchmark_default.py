import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "xgb_benchmark_numeric_default",
  "project": "thesis-3",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "maximize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "xgb_c"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["covertype",
                 "poker",
                 "MiniBooNE",
                 "Higgs",
                 "jannis"]
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
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-3")