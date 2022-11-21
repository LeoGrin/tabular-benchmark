import numpy as np

config_random = {"model_type": {
    "value": "sklearn"
},
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_depth": {
        "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    "model__learning_rate": {
        'distribution': "log_uniform_values",
        'min': 1E-5,  # inspired by RTDL
        'max': 0.7,
    },
    "model__n_estimators": {
        "value": 1_000,
        #"distribution": "q_uniform",
        #"min": 100,
        #"max": 6000,
        #"q": 200
    },
    "early_stopping_rounds": {
        "value": 20
    },
    "model__gamma": {
        "distribution": "log_uniform_values",
        'min': 1E-8,  # inspired by RTDL
        'max': 7,
    },
    "model__min_child_weight": {
        "distribution": "q_log_uniform_values",
        'min': 1,
        'max': 100,
        'q': 1
    },
    "model__subsample": {
        "distribution": "uniform",
        'min': 0.5,
        'max': 1.0
    },
    "model__colsample_bytree": {
        "distribution": "uniform",
        'min': 0.5,
        'max': 1.0
    },
    "model__colsample_bylevel": {
        "distribution": "uniform",
        'min': 0.5,
        'max': 1.0
    },
    "model__reg_alpha": {
        "distribution": "log_uniform_values",
        'min': 1E-8,  # inspired by RTDL
        'max': 1E2,
    },
    "model__reg_lambda": {
        "distribution": "log_uniform_values",
        'min': 1,
        'max': 4
    },
    "model__use_label_encoder": {
        "value": False
    },
    "transformed_target": {
        "values": [False, True]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
    "use_gpu": {
        "value": False
    }
}


config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
    "model__n_estimators": {
        "value": 1_000,
    },
    "early_stopping_rounds": {
        "value": 20
    },
}

config_regression = dict(config_random, **{
    "model_name": {
        "value": "xgb_r"
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "xgb_r"
    },
})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "xgb_c"
    },
})

config_classif_default= dict(config_default, **{
    "model_name": {
        "value": "xgb_c"
    },
})