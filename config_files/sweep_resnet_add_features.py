
import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "resnet_add_features",
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
      "value": "rtdl_resnet"
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
    "model__module__activation": {
      "value": "reglu"
    },
    "model__module__normalization": {
      "values": ["batchnorm", "layernorm"]
    },
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
      "values": ["heloc", "electricity", "california", "covtype", "churn", "cpu", "wine"]
    },
    "transform__0__method_name": {
      "value": "add_uninformative_features"
    },
    "transform__0__multiplier": {
      "values": [1., 1.5, 2],
    },
    "transform__1__method_name": {
      "value": "gaussienize"
    },
    "transform__1__type": {
      "value": "quantile",
    },
    "n_iter": {
      "value": 1,
    }
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")