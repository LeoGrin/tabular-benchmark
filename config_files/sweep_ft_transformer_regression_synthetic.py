import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "ft_transformer_default_regression_synthetic",
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
      "value": "ft_transformer_regressor"
    },
    "model__optimizer": {
      "value": "adamw"
    },
    "model__lr_scheduler": {
      "value": False
    },
    "model__batch_size": {
      "value": 512
    },
    "model__max_epochs": {
      "value": 300
    },
    "model__module__activation": {
      "value": "reglu"
    },
    "model__module__token_bias": {
      "value": True
    },
    "model__module__prenormalization": {
      "value": True
    },
    "model__module__kv_compression": {
      "value": True
    },
    "model__module__kv_compression_sharing": {
      "value": "headwise"
    },
    "model__module__initialization": {
      "value": "kaiming"
    },
    "model__module__n_layers": {
      "value": 3
    },
    "model__module__n_heads": {
      "value": 8,
    },
    "model__module__d_ffn_factor": {
      "value": 4./3
    },
    "model__module__ffn_dropout": {
      "value": 0.0
    },
    "model__module__attention_dropout": {
      "value": 0.0
    },
    "model__module__residual_dropout": {
      "value": 0.0
    },
    "model__lr": {
      "value": 1e-4,
    },
     "model__optimizer__weight_decay": {
      "value": 1e-5,
    },
    "d_token": {
      "value": 192
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
      "value": 5,
    },
    "regression": {
      "value": True
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")