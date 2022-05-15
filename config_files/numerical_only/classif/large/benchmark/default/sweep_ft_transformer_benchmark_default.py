import wandb
import numpy as np

sweep_config = {
    "program": "run_experiment.py",
    "name": "ft_transformer_benchmark_numeric_large",
    "project": "thesis",
    "method": "grid",
    "metric": {
        "name": "mean_test_score",
        "goal": "maximize"
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
            "value": "ft_transformer"
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
            "value": 4. / 3
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
