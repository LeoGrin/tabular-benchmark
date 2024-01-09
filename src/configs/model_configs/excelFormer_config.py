from configs.model_configs.skorch_config import skorch_config, skorch_config_default

# config_random = {
#     "model__module__activation": {
#         "value": "reglu"
#     },
#     "model__module__token_bias": {
#         "value": True
#     },
#     "model__module__prenormalization": {
#         "value": True
#     },
#     "model__module__kv_compression": {
#         "values": [True, False]
#     },
#     "model__module__kv_compression_sharing": {
#         "values": ["headwise", 'key-value']
#     },
#     "model__module__initialization": {
#         "value": "kaiming"
#     },
#     "model__module__n_layers": {
#         "distribution": "q_uniform",
#         "min": 1,
#         "max": 6
#     },
#     "model__module__n_heads": {
#         "value": 8,
#     },
#     "model__module__d_ffn_factor": {
#         "distribution": "uniform",
#         "min": 2. / 3,
#         "max": 8. / 3
#     },
#     "model__module__ffn_dropout": {
#         "distribution": "uniform",
#         "min": 0,
#         "max": 0.5
#     },
#     "model__module__attention_dropout": {
#         "distribution": "uniform",
#         "min": 0.0,
#         "max": 0.5
#     },
#     "model__module__residual_dropout": {
#         "distribution": "uniform",
#         "min": 0.0,
#         "max": 0.5
#     },
#     "model__lr": {
#         "distribution": "log_uniform_values",
#         "min": 1e-5,
#         "max": 1e-3
#     },
#     "model__optimizer__weight_decay": {
#         "distribution": "log_uniform_values",
#         "min": 1e-6,
#         "max": 1e-3
#     },
#     "d_token": { #modified in run_experiment.py
#         "distribution": "q_uniform",
#         "min": 64,
#         "max": 512
#     },
#     "model__lr_scheduler": {
#         "values": [True, False]
#     },
#     "use_gpu": {
#         "value": True
#     }
# }

config_default = {
    "model__module__token_bias": {
        "value": True
    },
    "model__module__beta": {
        "value": 0.5
    },
    "model__module__beta": {
        "value": 0.5
    },
    "model__module__mtype": {
        "value": "none"
    },
    "model__module__init_scale": {
        "value": 0.01
    },
    "model__lr_scheduler": {
        "value": False
    },
    #"model__module__activation": {
    #    "value": "reglu"
    #},
    "model__module__token_bias": {
        "value": True
    },
    "model__module__prenormalization": {
        "value": True
    },
    "module__kv_compression_temp": {
        "value": "None"
    },
    "module__kv_compression_sharing_temp": { #_temp means "None" is replaced by None
        "value": "None"
    },
    # "model__module__initialization": {
    #     "value": "kaiming"
    # },
    "model__module__n_layers": {
        "value": 3
    },
    "model__module__n_heads": {
        "value": 32,
    },
    # "model__module__d_ffn_factor": {
    #     "value": 4. / 3
    # },
    "model__module__ffn_dropout": {
        "value": 0.0
    },
    "model__module__attention_dropout": {
        "value": 0.3
    },
    "model__module__residual_dropout": {
        "value": 0.0
    },
    "model__lr": {
        "value": 1e-4,
    },
    "model__optimizer__weight_decay": {
        "value": 0.,
    },
    "model__module__d_token": { #TODO: do like FT_transformer
        "value": 256
    },
    "use_gpu": {
        "value": True
    }
}

# config_regression = dict(config_random,
#                                         **skorch_config,
#                                         **{
#                                             "model_name": {
#                                                 "value": "ft_transformer_regressor"
#                                             },
#                                         })

config_regression_default = dict(config_default,
                                        **skorch_config_default,
                                        **{
                                            "model_name": {
                                                "value": "excelFormer_regressor"
                                            },
                                        })

# config_classif = dict(config_random,
#                                      **skorch_config,
#                                      **{
#                                          "model_name": {
#                                              "value": "ft_transformer"
#                                          },
#                                      })

config_classif_default = dict(config_default,
                                     **skorch_config_default,
                                     **{
                                         "model_name": {
                                             "value": "excelFormer"
                                         },
                                     })