import numpy as np
import scipy.stats as stats
config = {
    "log_training": True,
    "model__device": "cuda",
    "model_type":  "skorch",
    "model_name": "rtdl_resnet",
    "model__optimizer": "adamw",
    "model__lr_scheduler": [True, False],
    "model__batch_size": [256, 512, 1024],
    "model__max_epochs": 300,
    "model__module__activation": "reglu",
    "model__module__normalization": ["batchnorm", "layernorm"],
    "model__module__n_layers": {
      "distribution": "q_uniform",
      "min": 1,
      "max": 16
    },
    "model__module__d": {
      "distribution": "q_uniform",
      "min": 64,
      "max": 1024
    },
    "model__module__d_hidden_factor": {
      "distribution": "uniform",
      "min": 1,
      "max": 4
    },
    "model__module__hidden_dropout": {
      "distribution": "uniform",
      "min": 0.0,
      "max": 0.5
    },
    "model__module__residual_dropout": {
      "distribution": "uniform",
      "min": 0.0,
      "max": 0.5
    },
    "model__lr": {
      "distribution": "log_uniform_values",
      "min": 1e-5,
      "max": 1e-2
    },
     "model__optimizer__weight_decay": {
      "distribution": "log_uniform_values",
      "min": 1e-8,
      "max": 1e-3
    },
    "model__module__d_embedding": {
      "distribution": "q_uniform",
      "min": 64,
      "max": 512
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["heloc", "electricity", "california", "covtype", "churn", "credit", "shopping", "nomao", "cpu", "spam", "wine"]
    },
    "transform__0__method_name": {
      "value": "gaussienize"
    },
    "transform__0__type": {
      "value": "quantile",
    },
    "n_iter": {
      "value": 1,
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")