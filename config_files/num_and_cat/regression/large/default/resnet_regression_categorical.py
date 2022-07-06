import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "resnet_regression_categorical_default",
  "project": "thesis-5",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters": {
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
      "values": ["black_friday",
                  "nyc-taxi-green-dec-2016",
                  "diamonds",
                  "Allstate_Claims_Severity",
                  "LoanDefaultPrediction",
                  "particulate-matter-ukair-2017",
                  "SGEMM_GPU_kernel_performance"]
    },
    "transform__0__method_name": {
      "value": "gaussienize"
    },
    "transform__0__type": {
      "value": "quantile",
    },
    "transform__0__apply_on": {
        "value": "numerical",
    },
    "n_iter": {
      "value": "auto",
    },
    "regression": {
          "value": True
    },
    "data__regression": {
          "value": True
    },
    "data__categorical": {
      "value": True
    },
    "transformed_target": {
      "values": [True]
    },
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-5")