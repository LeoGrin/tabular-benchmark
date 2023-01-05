import numpy as np
from configs.model_configs.skorch_config import skorch_config, skorch_config_default


config_random = {
    # parameters space used for random search
    # following WandB sweeps syntax
    "model_type": {
        "value": "sklearn" # or "skorch" or "tab_survey"
    },
    "model__parameter_name": { # parameters fed to the model constructor should be written model__parameter_name
        "value": 10
    },
    "model__parameter_name_with_multiple_values_to_try": { # to try during random search
        "values": [1, 2, 3]
    },
    "model__parameter_name_with_distribution": { #you can also specify a distribution. See WandB sweeps doc for more info on the syntax
        "distribution": "log_normal",
        'mu': float(np.log(0.01)),
        'sigma': float(np.log(10.0)),
    },
    "data__parameter": { # pass parameters to the data constructor
        "values": [1, 2, 3, "None"] # None doesn't work with wandb, "None" is converted to None
    },
    "transform__0__method_name": { # Optional: name of the first transform method to apply to X,y
        "value": "gaussienize"
    },
    "transform__0__parameter": { # Add argument to the transform function
        "value": "example_value",
    },
    "transform__0__apply_on": {
        "value": "numerical", #or categorical or both
    },
    "transformed_target": { # do you want the target to be gaussienized before fit and degaussienized before prediction
        "values": [False, True]
    },
    "use_gpu": {
        "value": False
    }
}

config_default = {
    # Default parameter values
    "model_type": {
        "value": "sklearn" # or "skorch" or "tab_survey"
    },
    "transformed_target": {
        "value": False
    },
    "use_gpu": {
        "value": False
    }

}

config_regression_random = dict(config_random, **{
    # add parameters space specific to regression situations
    "model_name": {
        "value": "model_name_r"
    },
    "model__loss": {
        "values": ["squared_error", "absolute_error"],
    },
})

config_regression_default = dict(config_default, **{
    # add default parameters specific to regression situations
    "model_name": {
        "value": "model_name_r"
    },
})

config_classif = dict(config_random, **{
    # add parameters space specific to classification situations
    "model_name": {
        "value": "model_name_c"
    },
})

config_classif_default = dict(config_default, **{
    # add default parameters specific to classification situations
    "model_name": {
        "value": "model_name_c"
    },
})