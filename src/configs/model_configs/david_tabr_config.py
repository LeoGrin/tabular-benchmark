
config_random  = {
    # specified for this benchmark
    "model__d_main": {
        "distribution": "q_uniform",
        "min": 16,
        "max": 384
    },
    "model__optimizer": {
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 3e-5,
                "max": 1e-3
            },
            "weight_decay": {
                #{0, LogUniform[1e-6, 1e-3]} #TODO: implement 0
                "distribution": "log_uniform_values",
                "min": 1e-9, #TODO: changed from 1e-6 to mimic 0 choice
                "max": 1e-4 #https://github.com/yandex-research/tabular-dl-tabr/blob/d628ec7e1c0a66011473021034e7dd4a77740112/exp/tabr/why/classif-cat-medium-0-compass/0-tuning/report.json
            }
        }
    },
    #encoder_n_blocks
    "model__encoder_n_blocks": {
        "values": [0, 1]
    },
    #predictor_n_blocks
    "model__predictor_n_blocks": {
        "values": [1, 2]
    },
    #dropout0
    #I think this correspond to FFN dropout in the paper
    "model__dropout0": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.6
    },
    #dropout1
    "model__dropout1": {
        #"value": "dropout0", #TODO: not sure about this
        "value": 0.0 #https://github.com/yandex-research/tabular-dl-tabr/blob/d628ec7e1c0a66011473021034e7dd4a77740112/exp/tabr/why/classif-cat-medium-0-compass/0-tuning/report.json
    },
    #context_dropout
    #I think this correspond to attention dropout in the paper
    "model__module__context_dropout": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.6
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
                                        "value": "david_tabr_regressor"
                                    },
                                }}

config_regression_default = {#**skorch_config_default,
                                 **config_default,
                                **{
                                    "model_name": {
                                        "value": "david_tabr_regressor"
                                    },
                                }}

config_classif = {#**skorch_config,
                      **config_random ,
                             **{
                                 "model_name": {
                                     "value": "david_tabr"
                                 },
                             }}

config_classif_default = {#**skorch_config_default,
                              **config_default,
                             **{
                                 "model_name": {
                                     "value": "david_tabr"
                                 },
                             }}
