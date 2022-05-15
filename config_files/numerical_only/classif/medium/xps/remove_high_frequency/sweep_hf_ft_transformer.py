import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "ft_transformer_hf_2",
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
      "value": "ft_transformer"
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
    "model__module__token_bias": {
      "value": True
    },
    "model__module__prenormalization": {
      "value": True
    },
    "model__module__kv_compression": {
      "values": [True, False]
    },
    "model__module__kv_compression_sharing": {
      "values": ["headwise", 'key-value']
    },
    "model__module__initialization": {
      "value": "kaiming"
    },
    "model__module__n_layers": {
      "distribution": "q_uniform",
      "min": 1,
      "max": 6
    },
    "model__module__n_heads": {
      "value": 8,
    },
    "model__module__d_ffn_factor": {
      "distribution": "uniform",
      "min": 2./3,
      "max": 8./3
    },
    "model__module__ffn_dropout": {
      "distribution": "uniform",
      "min": 0,
      "max": 0.5
    },
    "model__module__attention_dropout": {
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
      "max": 1e-3
    },
     "model__optimizer__weight_decay": {
      "distribution": "log_uniform_values",
      "min": 1e-6,
      "max": 1e-3
    },
    "d_token": {
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
    "transform__0__method_name": {
      "value": "gaussienize",
    },
    "transform__0__type": {
      "value": "quantile",
    },
    "transform__1__method_name": {
      "value": "remove_high_frequency_from_train",
    },
    "transform__1__cov_mult": {
      "values": [0, 0.001, 0.01, 0.1, 0.5]
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis")