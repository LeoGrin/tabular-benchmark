
# Default config for all skorch model
# Can be overwritten in the config file of a model

skorch_config = {
    "log_training": {
        "value": True
    },
    "model__device": {
        "value": "cuda"
    },
    "model_type": {
        "value": "skorch"
    },
    "model__use_checkpoints": {
        "value": True
    },
    "model__optimizer": {
        "value": "adamw"
    },
    "model__batch_size": {
        "values": [256, 512, 1024]
    },
    "model__max_epochs": {
        "value": 300
    },
    "transform__0__method_name": {
        "value": "gaussienize"
    },
    "transform__0__type": {
        "value": "quantile",
    },
    "transform__0__apply_on": {
        "value": "numerical",
    },
    "transformed_target": {
        "values": [False, True]
    },
}

skorch_config_default = skorch_config.copy()
skorch_config_default["model__batch_size"] = {"value": 512}
skorch_config_default["transformed_target"] = {"value": True}