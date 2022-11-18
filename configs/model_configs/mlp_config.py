from configs.model_configs.skorch_config import skorch_config, skorch_config_default

mlp_config = {
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
}

mlp_config_default = {
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
}

mlp_config_regression = dict(mlp_config,
                             **skorch_config,
                             **{
                                 "model_name": {
                                     "value": "rtdl_mlp_regressor"
                                 },
                             })

mlp_config_regression_default = dict(mlp_config_default,
                             **skorch_config_default,
                             **{
                                 "model_name": {
                                     "value": "rtdl_mlp_regressor"
                                 },
                             })

mlp_config_classif = dict(mlp_config,
                          **skorch_config,
                          **{
                              "model_name": {
                                  "value": "rtdl_mlp"
                              },
                          })

mlp_config_classif_default = dict(mlp_config_default,
                          **skorch_config_default,
                          **{
                              "model_name": {
                                  "value": "rtdl_mlp"
                              },
                          })
