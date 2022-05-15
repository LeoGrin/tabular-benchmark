import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "mlp_benchmark_numeric",
  "project": "thesis",
  "method" : "random",
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
      "values": [True, False]
    },
    "model__batch_size": {
      "values": [256, 512, 1024]
    },
    "model__max_epochs": {
      "value": 300
    },
    "model__module__n_layers": {
      "distribution": "q_uniform",
      "min": 1,
      "max": 8
    },
    "model__module__d_layers": {
      "distribution": "q_uniform",
      "min": 16,
      "max": 1024
    },
    "model__module__dropout": {
      "value": 0.0,
    },
    "model__lr": {
      "distribution": "log_uniform_values",
      "min": 1e-5,
      "max": 1e-2
    },
    #  "model__optimizer__weight_decay": {
    #   "distribution": "log_uniform_values",
    #   "min": 1e-8,
    #   "max": 1e-3
    # },
    "model__module__d_embedding": {
      "distribution": "q_uniform",
      "min": 64,
      "max": 512
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