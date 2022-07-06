import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "ft_transformer_default_numeric_regression",
  "project": "thesis-4",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
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
    "model__use_checkpoints": {
      "value": True
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
      "value": 0.1
    },
    "model__module__attention_dropout": {
      "value": 0.2
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
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["cpu_act",
                 "pol",
                 "elevators",
                 "isolet",
                 "wine_quality",
                  "Ailerons",
                 # "yprop_4_1",
                  "houses",
                  "house_16H",
                  #"delays_zurich_transport",
                  "diamonds",
                  "Brazilian_houses",
                  #"Allstate_Claims_Severity",
                  "Bike_Sharing_Demand",
                  #"OnlineNewsPopularity",
                  "nyc-taxi-green-dec-2016",
                  "house_sales",
                  "sulfur",
                  #"fps-in-video-games",
                  "medical_charges",
                  "MiamiHousing2016",
                  "superconduct",
                  "california",
                 "year",
                 "fifa"]
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
          "value": True
    },
    "data__regression": {
          "value": True
    },
    "transformed_target": {
        "value": True
    },
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-3")