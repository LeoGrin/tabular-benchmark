import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "xgb_benchmark_numeric_large",
  "project": "thesis",
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
      "value": "xgb_c"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_depth": {
      "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    "model__learning_rate": {
      'distribution': "log_uniform_values",
      'min': 1E-5, # inspired by RTDL
      'max': 0.7,
    },
    "model__n_estimators": {
      "distribution": "q_uniform",
      "min": 100,
      "max": 6000,
      "q": 200
    },
    "model__gamma": {
      "distribution": "log_uniform_values",
      'min': 1E-8, # inspired by RTDL
      'max': 7,
    },
    "model__min_child_weight": {
      "distribution": "q_log_uniform_values",
      'min': 1,
      'max': 100,
      'q': 1
    },
    "model__subsample": {
      "distribution": "uniform",
      'min': 0.5,
      'max': 1.0
    },
    "model__colsample_bytree": {
      "distribution": "uniform",
      'min': 0.5,
      'max': 1.0
    },
    "model__colsample_bylevel": {
      "distribution": "uniform",
      'min': 0.5,
      'max': 1.0
    },
    "model__reg_alpha": {
      "distribution": "log_uniform_values",
      'min': 1E-8, # inspired by RTDL
      'max': 1E2,
    },
    "model__reg_lambda": {
      "distribution": "log_uniform_values",
      'min': 1,
      'max': 4
    },
    "model__use_label_encoder": {
      "value": False
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
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")