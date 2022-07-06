import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "resnet_benchmark_numeric_large_default",
  "project": "thesis-3",
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
      "value": "rtdl_resnet"
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
    "model__module__activation": {
      "value": "reglu"
    },
    "model__module__normalization": {
      "values": ["batchnorm"]
    },
    "model__module__n_layers": {
      "value": 8,
    },
    "model__module__d": {
      "value": 256,
    },
    "model__module__d_hidden_factor": {
      "value": 2,
    },
    "model__module__hidden_dropout": {
      "value": 0.2,
    },
    "model__module__residual_dropout": {
      "value": 0.2
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
    "data__regression": {
      "value": False
    },
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-3")