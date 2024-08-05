config_random  = {
    "model__module__n_layers": {
        "values": list(range(1, 9))
    },
    "model__module__d_layers": {
        "values": list(range(1, 513))
    },
    "model__module__d_first_layer": {
        "values": list(range(1, 513))
    },
    "model__module__d_last_layer": {
        "values": list(range(1, 513))
    },
    "model__module__dropout": {
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
        "min": 1e-6,
        "max": 1e-3
    },
    "model__module__d_embedding": {
        "values": list(range(8, 32))
    },
    "model__module__num_emb_type": {
        "values": ['plr']
    },
    "model__module__num_emb_dim": {
        "values": list(range(1, 65))
    },
    "model__module__num_emb_hidden_dim": {
        "values": list(range(1, 65))
    },
    "model__module__num_emb_sigma": {
        "distribution": "log_uniform_values",
        "min": 1e-3,
        "max": 1e2
    },
    "model__batch_size": {
        "value": 256,
    },
    "model__optimizer": {
        "value": "adamw"
    },
    "model__max_epochs": {
        "value": 400
    },
    "model__use_checkpoints": {
        "value": True
    },
    "model__es_patience": {
        "value": 16
    },
    "model__verbose": {
        "value": 0
    },
    "model__tfms": {
        "values": ['quantile_tabr'],
    },
    "use_gpu": {
        "value": True
    },
    "model_type": {
        "value": "david"
    },
    "model__device": {
        "value": "cuda:0" #FIXME
    },
    "transformed_target": {
        "values": [False, True],
    },
    "transformed_target_type": {
        "value": "standard"
    },
}

config_default = {
    "use_gpu": {
        "value": True
    },
    "model_type": {
        "value": "david"
    },
    "model__device": {
        "value": "cuda:0" #FIXME
    },
    "transformed_target": {
        "value": False,
    },
}

config_regression = {#**skorch_config,
                         **config_random ,
                                **{
                                    "model_name": {
                                        "value": "david_mlp_plr_regressor"
                                    },
                                }}

config_regression_default = {#**skorch_config_default,
                                 **config_default,
                                **{
                                    "model_name": {
                                        "value": "david_mlp_plr_regressor"
                                    },
                                }}

config_classif = {#**skorch_config,
                      **config_random ,
                             **{
                                 "model_name": {
                                     "value": "david_mlp_plr"
                                 },
                             }}

config_classif_default = {#**skorch_config_default,
                              **config_default,
                             **{
                                 "model_name": {
                                     "value": "david_mlp_plr"
                                 },
                             }}
