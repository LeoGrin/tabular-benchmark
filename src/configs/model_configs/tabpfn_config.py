import numpy as np

config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "value": False
    },
    "transform__0__method_name": {
        "value": "ordinal"
    },
    "transform__0__apply_on": {
        "value": "categorical",
    },
    "use_gpu": {
        "value": False
    }
}

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "tabpfn"
    },
})
