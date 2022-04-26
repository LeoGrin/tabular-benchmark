import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "rotf_benchmark_numeric",
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
      "value": "rotation_forest"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__n_estimators": {
      "distribution": "q_log_uniform_values",
      "min": 10,
      "max": 1000,
      "q": 1
    },
    "model__n_features_per_subset": {
      "distribution": "q_uniform",
      "min": 3,
      "max": 12,
    },
    "model__rotation_algo": {
      "values": ["pca", "randomized"],
    },
    "model__criterion": {
      "values": ["gini", "entropy"],
    },
    "model__max_features": {
      "values": ["sqrt", "sqrt", "log2", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    },
    "model__max_depth": { # Not exactly like Hyperopt
      "values": [None, 2, 3, 5, 10, 15, 20],
    },
    "model__min_samples_split": {
      "value": 2
    },
    "model__min_samples_leaf": { # Not exactly like Hyperopt
      "distribution": "q_log_uniform_values",
      "min": 1.5,
      "max": 50.5,
      "q": 1
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["heloc", "electricity", "california", "covtype", "churn", "cpu", "wine"]
    },
    "n_iter": {
      "value": 1,
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")