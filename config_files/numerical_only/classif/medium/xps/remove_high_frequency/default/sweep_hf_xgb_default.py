import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "xgb_benchmark_numeric_default",
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
      "value": "xgb_c"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 "covertype",
                 "poker",
                 "pol",
                 "house_16H",
                 "kdd_ipums_la_97-small",
                 "MagicTelescope",
                 "bank-marketing",
                 "phoneme",
                 "MiniBooNE",
                 "Higgs",
                 "eye_movements",
                 "jannis",
                 "credit",
                 "california",
                 "wine"]
    },
      "transform__0__method_name": {
          "value": "gaussienize",
      },
      "transform__0__type": {
          "value": "quantile",
      },
      "transform__1__method_name": {
          "value": "remove_high_frequency_from_train",
      },
      "transform__1__cov_mult": {
          "values": [0, 0.001, 0.01, 0.1, 0.5]
      },
      "n_iter": {
          "value": "auto",
      },
      "regression": {
          "value": False
      },
      "max_train_samples": {
          "value": 10000
      },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")