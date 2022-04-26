import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "rf_benchmark_numeric_default",
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
      "value": "rf_c"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"]
    },
    "n_iter": {
      "value": 1,
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")