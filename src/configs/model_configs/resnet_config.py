from configs.model_configs.skorch_config import skorch_config, skorch_config_default

config_random  = {
    "model__module__activation": {
        "value": "reglu"
    },
    "model__module__normalization": {
        "values": ["batchnorm", "layernorm"]
    },
    "model__module__n_layers": {
        "distribution": "q_uniform",
        "min": 1,
        "max": 16
    },
    "model__module__d": {
        "distribution": "q_uniform",
        "min": 64,
        "max": 1024
    },
    "model__module__d_hidden_factor": {
        "distribution": "uniform",
        "min": 1,
        "max": 4
    },
    "model__module__hidden_dropout": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5
    },
    "model__module__residual_dropout": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5
    },
    "model__lr": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-2
    },
    "model__optimizer__weight_decay": {
        "distribution": "log_uniform_values",
        "min": 1e-8,
        "max": 1e-3
    },
    "model__module__d_embedding": {
        "distribution": "q_uniform",
        "min": 64,
        "max": 512
    },
    "model__lr_scheduler": {
        "values": [True, False]
    },
}
config_default = {
    "model__lr_scheduler": {
        "values": [True]
    },
    "model__module__activation": {
        "value": "reglu"
    },
    "model__module__normalization": {
        "values": ["batchnorm"]
    },
    "model__module__n_layers": {
        "value": 8,
    },
    "model__module__d": {
        "value": 256,
    },
    "model__module__d_hidden_factor": {
        "value": 2,
    },
    "model__module__hidden_dropout": {
        "value": 0.2,
    },
    "model__module__residual_dropout": {
        "value": 0.2
    },
    "model__lr": {
        "value": 1e-3,
    },
    "model__optimizer__weight_decay": {
        "value": 1e-7,
    },
    "model__module__d_embedding": {
        "value": 128
    },
    "use_gpu": {
        "value": True
    }
}

config_regression = dict(config_random ,
                                **skorch_config,
                                **{
                                    "model_name": {
                                        "value": "rtdl_resnet_regressor"
                                    },
                                })

config_regression_default = dict(config_default,
                                **skorch_config_default,
                                **{
                                    "model_name": {
                                        "value": "rtdl_resnet_regressor"
                                    },
                                })

config_classif = dict(config_random ,
                             **skorch_config,
                             **{
                                 "model_name": {
                                     "value": "rtdl_resnet"
                                 },
                                "model__early_stop_on": {
                                  "valid_acc",
                              }
                             })

config_classif_default = dict(config_default,
                             **skorch_config_default,
                             **{
                                 "model_name": {
                                     "value": "rtdl_resnet"
                                 },
                                "model__early_stop_on": {
                                  "valid_acc",
                              }
                             })