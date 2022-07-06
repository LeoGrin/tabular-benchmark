import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "saint_numerical_regression_large",
  "project": "thesis-4",
  "method" : "random",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters" : {
      "model__args__lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3
      },
      "model__args__batch_size": {
          "values":  [128, 256],
      },
      "model__args__val_batch_size":{
          "values": [128, 256],
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
          "value": "regression",
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
          "values": [1, 2, 3, 6, 12],
      },
      "model__params__heads": {
          "values": [2, 4, 8],
      },
    "model__params__dim": {
      "values": [32, 64, 128]#, 256],
    },
      "model__params__dropout": {
          "values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["diamonds",
                  "nyc-taxi-green-dec-2016",
                 "year"]
    },
    "transform__0__method_name": {
      "value": "gaussienize"
    },
    "transform__0__type": {
      "value": "quantile",
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


sweep_id = wandb.sweep(sweep_config, project="thesis-4")