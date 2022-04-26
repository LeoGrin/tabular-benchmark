import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "rf_default_regression_synthetic",
  "project": "thesis",
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
      "value": "rf_r"
    },
    "data__method_name": {
      "value": "uniform_data"
    },
    "data__n_samples": {
      "values": [10000]
    },
    "data__n_features": {
      "value": 1,
    },
    "max_train_samples": {
      "values": [500, 1000]
    },
    "target__method_name": {
      "value": "periodic_triangle"
    },
    "target__n_periods": {
      "value": 8
    },
    "target__period_size": {
      "values": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    },
    "target__noise": {
      "value": False
    },
    "transform__0__method_name": {
      "value": "gaussienize"
    },
    "transform__0__type": {
      "values": ["quantile", "identity"],
    },
    "n_iter": {
      "value": 10,
    },
    "regression": {
      "value": True
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")