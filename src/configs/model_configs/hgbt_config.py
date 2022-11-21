import numpy as np

config_random = {
    "model_type": {
        "value": "sklearn"
    },
    "model__max_iter": {
        "value": 1_000
    },
    "model__early_stopping": {
        "value": True
    },
    "model__validation_fraction": {
        "value": 0.2
    },
    "model__n_iter_no_change": {
        "value": 20
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__learning_rate": {
        'distribution': "log_normal",
        'mu': float(np.log(0.01)),
        'sigma': float(np.log(10.0)),
    },
    "model__max_leaf_nodes": {
        'distribution': "q_normal",
        'mu': 31,
        "sigma": 5
    },
    "max_depth_temp": {  # Added None compared to hyperopt
        "values": ["None", 2, 3, 4],
        "probabilities": [0.1, 0.1, 0.7, 0.1]
    },
    "model__min_samples_leaf": {  # Not exactly like Hyperopt
        "distribution": "q_normal",
        "mu": 20,
        "sigma": 2,
        "q": 1
    },
    "transformed_target": {
        "values": [False, True]
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
    "model__max_iter": {
        "value": 1_000
    },
    "model__early_stopping": {
        "value": True
    },
    "model__validation_fraction": {
        "value": 0.2
    },
    "model__n_iter_no_change": {
        "value": 20
    },
}

config_regression = dict(config_random, **{
    "model_name": {
        "value": "hgbt_r"
    },
    "model__loss": {
        "values": ["squared_error", "absolute_error"],
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "hgbt_r"
    },
})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "hgbt_c"
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "hgbt_c"
    },
})