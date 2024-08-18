import numpy as np

config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "value": False
    },
    "one_hot_encoder": { # Use one-hot encoding for categorical variables when needed
        "value": True
    },
    "use_gpu": {
        "value": False
    }
}

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "ensemble_mothernet"
    },
})
