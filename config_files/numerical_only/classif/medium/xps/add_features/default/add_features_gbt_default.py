import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "add_features_gbt_default",
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
      "value": "gbt_c"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 "covertype",
                 # "poker",
                 "pol",
                 "house_16H",
                 "kdd_ipums_la_97-small",
                 "MagicTelescope",
                 "bank-marketing",
                 "phoneme",
                 "MiniBooNE",
                 "Higgs",
                 "eye_movements",
                 # "jannis",
                 "credit",
                 "california",
                 "wine"]
    },
    "transform__0__method_name": {
    "value": "add_uninformative_features"
    },
    "transform__0__multiplier": {
    "values": [1., 1.5, 2],
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