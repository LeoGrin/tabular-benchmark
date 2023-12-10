
config_random  = {
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
    "transform__0__method_name": {
        "value": "normalize_y"
    },
    "transform__0__apply_on": {
        "value": "all",
    },

}

config_regression = {#**skorch_config,
                         **config_random ,
                                **{
                                    "model_name": {
                                        "value": "david_regressor_normalized_y"
                                    },
                                }}

config_regression_default = {#**skorch_config_default,
                                 **config_default,
                                **{
                                    "model_name": {
                                        "value": "david_regressor_normalized_y"
                                    },
                                }}

config_classif = {#**skorch_config,
                      **config_random ,
                             **{
                                 "model_name": {
                                     "value": "david_normalized_y"
                                 },
                             }}

config_classif_default = {#**skorch_config_default,
                              **config_default,
                             **{
                                 "model_name": {
                                     "value": "david_normalized_y"
                                 },
                             }}