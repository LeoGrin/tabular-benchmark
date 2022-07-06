import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "xgb_regression_categorical_large",
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
      "value": "xgb_r"
    },
    "model__use_label_encoder": {
      "value": False
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
    "one_hot_encoder": {
        "value": True
    },
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-5")