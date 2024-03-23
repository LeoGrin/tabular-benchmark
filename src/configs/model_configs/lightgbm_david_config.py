
config_random = {
    "model__lr": {
        'distribution': "log_uniform",
        'min': -7,
        'max': 0,
    },
    "model__num_leaves": {
        'distribution': "q_log_uniform",
        'min': 0,
        'max': 7,
        'q': 1
    },
    "model__colsample_bytree": { #feature fraction
        'distribution': "uniform",
        'min': 0.5,
        'max': 1.0,
    },
    "model__subsample": { #bagging fraction
        'distribution': "uniform",
        'min': 0.5,
        'max': 1.0,
    },
    "model__min_data_in_leaf": {
        'distribution': "q_log_uniform",
        'min': 0,
        'max': 6,
        'q': 1
    },
    "model__min_sum_hessian_in_leaf": {
        'distribution': "log_uniform",
        'min': -16,
        'max': 5,
    },
    "model__lambda_l1": {
        'distribution': "categorical",
        'values': [0, {'distribution': "log_uniform", 'min': -16, 'max': 2}],
    },
    "model__lambda_l2": {
        'distribution': "categorical",
        'values': [0, {'distribution': "log_uniform", 'min': -16, 'max': 2}],
    },
    "model__n_estimators": {
        "value": 1000
    },
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
    },
    "model__n_threads": {
        "value": 1,
    }
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
    },
    "model__n_threads": {
        "value": 1,
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