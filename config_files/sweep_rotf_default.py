import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "rotf_numeric_default",
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
      "value": "rotation_forest"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["heloc", "electricity", "california", "covtype", "churn", "cpu", "wine"]
    },
    "regression": {
      "value": False
    },
    "n_iter": {
      "value": 1,
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")