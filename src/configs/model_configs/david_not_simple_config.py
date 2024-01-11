
config_random  = {
    "model__num_emb_type": {
        "values": ['none', 'pl-densenet', 'plr']
    },
    "model__use_front_scale": {
        "values": [True, False],
        "probabilities": [0.6, 0.4]
    },
    'model__lr': {
        "distribution": "log_uniform_values",
        "min": 2e-2,
        "max": 1.5e-1,
    },
    'model__p_drop': {
            'values': [0.0, 0.15, 0.3, 0.45],
            'probabilities': [0.2, 0.4, 0.2, 0.2]
    },
    'model__act': {
            'values': ['relu', 'selu', 'mish']
    },
    'model__hidden_sizes': {
            'values': [[256] * 3, [64] * 3, [512]],
            'probabilities': [0.6, 0.2, 0.2]
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

#Defaults for TabR-S
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
                                        "value": "david_not_simple_regressor"
                                    },
                                    "model__wd": {
                                        "values": [0.0, 1e-3]
                                    }
                                }}

config_regression_default = {#**skorch_config_default,
                                 **config_default,
                                **{
                                    "model_name": {
                                        "value": "david_not_simple_regressor"
                                    },
                                }}

config_classif = {#**skorch_config,
                      **config_random ,
                             **{
                                 "model_name": {
                                     "value": "david_not_simple"
                                 },
                                "model__wd": {
                                    "values": [0.0, 1e-3, 1e-2]
                                 },
                                 "model__ls_eps": {
                                    "values": [0.0, 0.1]
                                 }
                             }}

config_classif_default = {#**skorch_config_default,
                              **config_default,
                             **{
                                 "model_name": {
                                     "value": "david_not_simple"
                                 },
                             }}