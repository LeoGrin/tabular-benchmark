import wandb

sweep_config = {
  "program": "run_experiment.py",
  "name" : "ft_transformer_regression_categorical",
  "project": "thesis-3",
  "method" : "random",
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
      "values": ["yprop_4_1",
                  "analcatdata_supreme",
                  "visualizing_soil",
                  "black_friday",
                  "nyc-taxi-green-dec-2016",
                  "diamonds",
                  "Allstate_Claims_Severity",
                  "Mercedes_Benz_Greener_Manufacturing",
                  "Brazilian_houses",
                  "Bike_Sharing_Demand",
                  "OnlineNewsPopularity",
                  "house_sales",
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
      "values": [False, True]
    },
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-3")