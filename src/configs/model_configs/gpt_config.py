import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


config_random = {"model_type": {
    "value": "sklearn"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__learning_rate": {
        'distribution': "log_normal",
        'mu': float(np.log(0.01)),
        'sigma': float(np.log(10.0)),
    },
    "model__subsample": {  # Not exactly like Hyperopt-sklearn
        'distribution': "uniform",
        'min': 0.5,
        'max': 1.0,
    },
    "model__n_estimators": {
        "value": 1_000 # Changed as asked by the reviewer
        # "distribution": "q_log_uniform_values",
        # "min": 10.5,
        # "max": 1000.5,
        # "q": 1
    },
    "model__n_iter_no_change": {
        "value": 20
    },
    "model__validation_fraction": {
        "value": 0.2
    },
    "model__criterion": {
        "values": ["friedman_mse", "squared_error"]
    },
    "max_depth_temp": {  # Not exactly like Hyperopt
        "values": ["None", 2, 3, 4, 5],
        "probabilities": [0.1, 0.1, 0.6, 0.1, 0.1]
    },
    "model__min_samples_split": {
        "values": [2, 3],
        "probabilities": [0.95, 0.05]
    },
    "model__min_samples_leaf": {  # Not exactly like Hyperopt
        "distribution": "q_log_uniform_values",
        "min": 1.5,
        "max": 50.5,
        "q": 1
    },
    "model__min_impurity_decrease": {
        "values": [0.0, 0.01, 0.02, 0.05],
        "probabilities": [0.85, 0.05, 0.05, 0.05],
    },
    "max_leaf_nodes_temp": {
        "values": ["None", 5, 10, 15],
        "probabilities": [0.85, 0.05, 0.05, 0.05]
    },
    "transformed_target": {
        "values": [False, True]
    },
    "one_hot_encoder": { # Use one-hot encoding for categorical variables when needed
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
     "use_gpu": {
        "value": False
    }
}

config_regression = dict(config_random, **{
    "model_name": {
        "value": "gbt_r"
    },
    "model__loss": {
        "values": ["squared_error", "absolute_error", "huber"],
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "gbt_r"
    },
})


config_classif = dict(config_random, **{
    "model_name": {
        "value": "gbt_c"
    },
    "model__loss": {
        "values": ["deviance", "exponential"],
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "gbt_c"
    },
})
