
config_random  = {
}
#Defaults for TabR-S
config_default = {
    "use_gpu": {
        "value": False
    },
    "model_type": {
        "value": "david"
    },
    "model__device": {
        "value": "cpu" #FIXME
    },
    "transformed_target": {
        "value": False,
    }
}

config_regression = {#**skorch_config,
                         **config_random ,
                                **{
                                    "model_name": {
                                        "value": "david_lightgbm_regressor"
                                    },
                                }}

config_regression_default = {#**skorch_config_default,
                                 **config_default,
                                **{
                                    "model_name": {
                                        "value": "david_lightgbm_regressor"
                                    },
                                }}

config_classif = {#**skorch_config,
                      **config_random ,
                             **{
                                 "model_name": {
                                     "value": "david_lightgbm"
                                 },
                             }}

config_classif_default = {#**skorch_config_default,
                              **config_default,
                             **{
                                 "model_name": {
                                     "value": "david_lightgbm"
                                 },
                             }}