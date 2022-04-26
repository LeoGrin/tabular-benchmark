import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "resnet_default_regression_synthetic",
  "project": "thesis",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "maximize"
  },
  "parameters" : {
    "log_training": {
      "value": True
    },
    "model__device": {
      "value": "cuda"
    },
    "model_type": {
      "value": "skorch"
    },
    "model_name": {
      "value": "rtdl_resnet_regressor"
    },
    "model__optimizer": {
      "value": "adamw"
    },
    "model__lr_scheduler": {
      "values": [True]
    },
    "model__batch_size": {
      "values": [512]
    },
    "model__max_epochs": {
      "value": 300
    },
    "model__module__activation": {
      "value": "reglu"
    },
    "model__module__normalization": {
      "values": ["batchnorm"]
    },
    "model__module__n_layers": {
      "values": [3, 8],
    },
    "model__module__d": {
      "values": [256, 1024],
    },
    "model__module__d_hidden_factor": {
      "value": 2,
    },
    "model__module__hidden_dropout": {
      "value": 0.0,
    },
    "model__module__residual_dropout": {
      "value": 0.0
    },
    "model__lr": {
      "value": 1e-3,
    },
     "model__optimizer__weight_decay": {
      "value": 1e-7,
    },
    "model__module__d_embedding": {
      "value": 128
    },
    "data__method_name": {
      "value": "uniform_data"
    },
    "data__n_samples": {
      "values": [20000]
    },
    "data__n_features": {
      "value": 1,
    },
    "max_train_samples": {
      "values": [10000]
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