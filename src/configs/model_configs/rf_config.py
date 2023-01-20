import numpy as np

config_random = {
    "model_type": {
        "value": "sklearn"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__n_estimators": {
        "value": 250,
        # "distribution": "q_log_uniform_values",
        # "min": 9.5,
        # "max": 3000.5,
        # "q": 1
    },
    "max_features_temp": {  # like Hyperopt ?
        "values": ["sqrt", "sqrt", "log2", "None", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    "max_depth_temp": {  # Not exactly like Hyperopt
        "values": ["None", 2, 3, 4],
        "probabilities": [0.7, 0.1, 0.1, 0.1]
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
    "model__bootstrap": {
        "values": [True, False]
    },
    "model__min_impurity_decrease": {
        "values": [0.0, 0.01, 0.02, 0.05],
        "probabilities": [0.85, 0.05, 0.05, 0.05]
    },
    "transformed_target": {
        "values": [False, True]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
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
        "value": "rf_r"
    },
    "model__criterion": {
        "values": ["squared_error", "absolute_error"],
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "rf_r"
    },

})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "rf_c"
    },
    "model__criterion": {
        "values": ["gini", "entropy"],
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "rf_c"
    },
})
