import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "gbt_categorical_classif_default",
  "project": "thesis-4",
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
      "value": "gbt_c"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 "eye_movements",
                  "KDDCup09_upselling",
                  "covertype",
                  "rl",
                  "road-safety",
                  "compass"]
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
    "data__categorical": {
            "value": True
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