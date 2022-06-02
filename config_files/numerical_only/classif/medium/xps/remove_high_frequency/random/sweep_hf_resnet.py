
import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "resnet_hf_new_005",
  "project": "thesis-2",
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
      "values": ["electricity",
                 "covertype",
                 "poker",
                 "pol",
                 "house_16H",
                 "kdd_ipums_la_97-small",
                 "MagicTelescope",
                 "bank-marketing",
                 "phoneme",
                 "MiniBooNE",
                 "Higgs",
                 "eye_movements",
                 "jannis",
                 "credit",
                 "california",
                 "wine"]
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
      "value": 10000
    },
    "transform__0__method_name": {
      "value": "gaussienize",
    },
    "transform__0__type": {
      "value": "quantile",
    },
    "transform__1__method_name": {
      "value": "select_features_rf",
    },
    "transform__1__num_features": {
      "value": 5,
    },
    "transform__2__method_name": {
      "value": "remove_high_frequency_from_train",
    },
    "transform__2__cov_mult": {
      "values": [0.05]
    },
    "transform__2__covariance_estimation": {
      "values": ["robust"]
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-2")