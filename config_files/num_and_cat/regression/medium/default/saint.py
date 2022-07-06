import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "saint_regression_categorical_medium_comparison_default",
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
      "values": ["black_friday",
                  "nyc-taxi-green-dec-2016",
                  "diamonds",
                  "Allstate_Claims_Severity",
                  "LoanDefaultPrediction",
                  "particulate-matter-ukair-2017",
                  "SGEMM_GPU_kernel_performance"]
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
          "value": True
    },
    "data__regression": {
          "value": True
    },
    "data__categorical": {
      "value": True
    },
    "transformed_target": {
      "values": [False]
    },
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-5")