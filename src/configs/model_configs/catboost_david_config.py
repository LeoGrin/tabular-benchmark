import numpy as np

# from Shwartz-Ziv and Armon, Tabular data: Deep learning is not all you need
# same as NODE except higher upper bound for leaf estimation iterations
# the parameter names in the space are for the alg interface, not directly for the GBDT interface!
config_random = {
    "model__lr": {
        'distribution': "log_uniform",
        'min': -5,
        'max': 0,
    },
    "model__random_strength": {
        'distribution': "q_uniform",
        'min': 1,
        'max': 20,
        'q': 1
    },
    "model__one_hot_max_size": {
        'distribution': "q_uniform",
        'min': 0,
        'max': 25,
        'q': 1
    },
    "model__l2_leaf_reg": {
        'distribution': "log_uniform_values",
        'min': 1,
        'max': 10,
    },
    "model__bagging_temperature": {
        'distribution': "uniform",
        'min': 0,
        'max': 1,
    },
    "model__bootstrap_type": {
        "value": "Bayesian",
    },
    "model__leaf_estimation_iterations": {
        'distribution': "q_uniform",
        'min': 1,
        'max': 20,
        'q': 1
    },
    "model__n_estimators": {
        "value": 2048
    },
    "model__max_depth": {
        "value": 6
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
                                        "value": "david_catboost_regressor"
                                    },
                                }}

config_regression_default = {#**skorch_config_default,
                                 **config_default,
                                **{
                                    "model_name": {
                                        "value": "david_catboost_regressor"
                                    },
                                }}

config_classif = {#**skorch_config,
                      **config_random ,
                             **{
                                 "model_name": {
                                     "value": "david_catboost"
                                 },
                             }}

config_classif_default = {#**skorch_config_default,
                              **config_default,
                             **{
                                 "model_name": {
                                     "value": "david_catboost"
                                 },
                             }}