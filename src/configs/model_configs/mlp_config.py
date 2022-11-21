from configs.model_configs.skorch_config import skorch_config, skorch_config_default

config_random  = {
    "model__module__n_layers": {
        "distribution": "q_uniform",
        "min": 1,
        "max": 8
    },
    "model__module__d_layers": {
        "distribution": "q_uniform",
        "min": 16,
        "max": 1024
    },
    "model__module__dropout": {
        "value": 0.0,
    },
    "model__lr": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-2
    },
    "model__module__d_embedding": {
        "distribution": "q_uniform",
        "min": 64,
        "max": 512
    },
    "model__lr_scheduler": {
        "values": [True, False]
    },
    "use_gpu": {
        "value": True
    }
}

config_default = {
    "model__lr_scheduler": {
        "values": [True]
    },
    "model__module__n_layers": {
        "value": 4,
    },
    "model__module__d_layers": {
        "value": 256,
    },
    "model__module__dropout": {
        "value": 0.0
    },
    "model__lr": {
        "value": 1e-3,
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
                                     "value": "rtdl_mlp_regressor"
                                 },
                             })

config_regression_default = dict(config_default,
                             **skorch_config_default,
                             **{
                                 "model_name": {
                                     "value": "rtdl_mlp_regressor"
                                 },
                             })

config_classif = dict(config_random ,
                          **skorch_config,
                          **{
                              "model_name": {
                                  "value": "rtdl_mlp"
                              },
                          })

config_classif_default = dict(config_default,
                          **skorch_config_default,
                          **{
                              "model_name": {
                                  "value": "rtdl_mlp"
                              },
                          })
