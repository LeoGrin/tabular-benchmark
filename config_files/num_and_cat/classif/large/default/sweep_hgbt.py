import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "hgbt_categorical_classif_large_default",
  "project": "thesis-5",
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
      "value": "hgbt_c"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 #"eye_movements",
                 # "KDDCup09_upselling",
                  "covertype",
                 # "rl",
                  "road-safety"]
                  #"compass"]
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
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-5")