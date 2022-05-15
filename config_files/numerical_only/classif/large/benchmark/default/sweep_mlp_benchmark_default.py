import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "mlp_benchmark_numeric_large",
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
      "value": "rtdl_mlp"
    },
    "model__use_checkpoints": {
      "value": True
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
    "model__module__n_layers": {
      "value": 4,
    },
    "model__module__d_layers": {
      "value": 256,
    },
    "model__module__dropout": {
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
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["covertype",
                 "poker",
                 "MiniBooNE",
                 "Higgs",
                 "jannis"]
    },
    "transform__0__method_name": {
      "value": "gaussienize"
    },
    "transform__0__type": {
      "value": "quantile",
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