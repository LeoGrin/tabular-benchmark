import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "saint_classif_categorical_large_default",
  "project": "thesis-5",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "maximize"
  },
    "parameters": {
        "model__args__lr": {
            "value": float(3e-5),
        },
        "model__args__batch_size": {
            "values": [128],
        },
        "model__args__val_batch_size": {
            "values": [128],
        },
        "model__args__epochs": {
            "value": 300,
        },
        "model__args__early_stopping_rounds": {
            "value": 10,
        },
        "model__args__use_gpu": {
            "value": True,
        },
        "model__args__data_parallel": {
            "value": False,
        },
        "model__args__num_classes": {
            "value": 1,
        },
        "model__args__objective": {
            "value": "binary",
        },
        "model__args__model_name": {
            "value": 'saint',
        },
        "model_name": {
            "value": "saint",
        },
        "model_type": {
            "value": "tab_survey",
        },
        "model__params__depth": {
            "values": [3],
        },
        "model__params__heads": {
            "values": [4],
        },
        "model__params__dim": {
            "values": [128],
        },
        "model__params__dropout": {
            "values": [0.1],
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
    "transform__0__method_name": {
      "value": "gaussienize"
    },
    "transform__0__type": {
      "value": "quantile",
    },
    "transform__0__apply_on": {
      "value": "numerical",
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