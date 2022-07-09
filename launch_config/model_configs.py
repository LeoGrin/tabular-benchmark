import numpy as np

# Default hyperparameters space for all models
# Taken from Hyperopt-sklearn or from the model's original paper

#########################################################
# Tree based models
#########################################################

############################
# Gradient Boosting Trees
############################

gbt_config = {"model_type": {
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
        "distribution": "q_log_uniform_values",
        "min": 10.5,
        "max": 1000.5,
        "q": 1
    },
    "model__criterion": {
        "values": ["friedman_mse", "squared_error"]
    },
    "model__max_depth": {  # Not exactly like Hyperopt
        "values": [None, 2, 3, 4, 5],
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
    "model__max_leaf_nodes": {
        "values": [None, 5, 10, 15],
        "probabilities": [0.85, 0.05, 0.05, 0.05]
    },
    "transformed_target": {
        "values": [False, True]
    },
    "one_hot_encoder": { # Use one-hot encoding for categorical variables when needed
        "value": True
    },
}

gbt_config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
}

gbt_config_regression = dict(gbt_config, **{
    "model_name": {
        "value": "gbt_r"
    },
    "model__loss": {
        "values": ["squared_error", "absolute_error", "huber"],
    },
})

gbt_config_regression_default = dict(gbt_config_default, **{
    "model_name": {
        "value": "gbt_r"
    },
})


gbt_config_classif = dict(gbt_config, **{
    "model_name": {
        "value": "gbt_c"
    },
    "model__loss": {
        "values": ["deviance", "exponential"],
    },
})

gbt_config_classif_default = dict(gbt_config_default, **{
    "model_name": {
        "value": "gbt_c"
    },
})

############################
# Random Forest
############################
rf_config = {
    "model_type": {
        "value": "sklearn"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__n_estimators": {
        "distribution": "q_log_uniform_values",
        "min": 9.5,
        "max": 3000.5,
        "q": 1
    },
    "model__criterion": {
        "values": ["squared_error", "absolute_error"],
    },
    "model__max_features": {  # like Hyperopt ?
        "values": ["sqrt", "sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    "model__max_depth": {  # Not exactly like Hyperopt
        "values": [None, 2, 3, 4],
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

rf_config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
}

rf_config_regression = dict(rf_config, **{
    "model_name": {
        "value": "rf_r"
    },
})

rf_config_regression_default = dict(rf_config_default, **{
    "model_name": {
        "value": "rf_r"
    },
})

rf_config_classif = dict(rf_config, **{
    "model_name": {
        "value": "rf_c"
    },
})

rf_config_classif_default = dict(rf_config_default, **{
    "model_name": {
        "value": "rf_c"
    },
})

############################
# XGBoost
############################
xgb_config = {"model_type": {
    "value": "sklearn"
},
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_depth": {
        "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    "model__learning_rate": {
        'distribution': "log_uniform_values",
        'min': 1E-5,  # inspired by RTDL
        'max': 0.7,
    },
    "model__n_estimators": {
        "distribution": "q_uniform",
        "min": 100,
        "max": 6000,
        "q": 200
    },
    "model__gamma": {
        "distribution": "log_uniform_values",
        'min': 1E-8,  # inspired by RTDL
        'max': 7,
    },
    "model__min_child_weight": {
        "distribution": "q_log_uniform_values",
        'min': 1,
        'max': 100,
        'q': 1
    },
    "model__subsample": {
        "distribution": "uniform",
        'min': 0.5,
        'max': 1.0
    },
    "model__colsample_bytree": {
        "distribution": "uniform",
        'min': 0.5,
        'max': 1.0
    },
    "model__colsample_bylevel": {
        "distribution": "uniform",
        'min': 0.5,
        'max': 1.0
    },
    "model__reg_alpha": {
        "distribution": "log_uniform_values",
        'min': 1E-8,  # inspired by RTDL
        'max': 1E2,
    },
    "model__reg_lambda": {
        "distribution": "log_uniform_values",
        'min': 1,
        'max': 4
    },
    "model__use_label_encoder": {
        "value": False
    },
    "transformed_target": {
        "values": [False, True]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
}


xgb_config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
}

xgb_config_regression = dict(xgb_config, **{
    "model_name": {
        "value": "xgb_r"
    },
})

xgb_config_regression_default = dict(xgb_config_default, **{
    "model_name": {
        "value": "xgb_r"
    },
})

xgb_config_classif = dict(xgb_config, **{
    "model_name": {
        "value": "xgb_c"
    },
})

xgb_config_classif_default= dict(xgb_config_default, **{
    "model_name": {
        "value": "xgb_c"
    },
})

############################
# Hist Gradient Boosting
############################
hgbt_config = {
    "model_type": {
        "value": "sklearn"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__learning_rate": {
        'distribution': "log_normal",
        'mu': float(np.log(0.01)),
        'sigma': float(np.log(10.0)),
    },
    "model__max_leaf_nodes": {
        'distribution': "q_normal",
        'mu': 31,
        "sigma": 5
    },
    "model__max_depth": {  # Added None compared to hyperopt
        "values": [None, 2, 3, 4],
        "probabilities": [0.1, 0.1, 0.7, 0.1]
    },
    "model__min_samples_leaf": {  # Not exactly like Hyperopt
        "distribution": "q_normal",
        "mu": 20,
        "sigma": 2,
        "q": 1
    },
    "transformed_target": {
        "values": [False, True]
    },
}

hgbt_config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
}

hgbt_config_regression = dict(hgbt_config, **{
    "model_name": {
        "value": "hgbt_r"
    },
    "model__loss": {
        "values": ["squared_error", "absolute_error"],
    },
})

hgbt_config_regression_default = dict(hgbt_config_default, **{
    "model_name": {
        "value": "hgbt_r"
    },
})

hgbt_config_classif = dict(hgbt_config, **{
    "model_name": {
        "value": "hgbt_c"
    },
})

hgbt_config_classif_default = dict(hgbt_config_default, **{
    "model_name": {
        "value": "hgbt_c"
    },
})

#########################################################
# Neural networks
#########################################################

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

############################
# FT Transformer
############################

ft_transformer_config = {
    "model__module__activation": {
        "value": "reglu"
    },
    "model__module__token_bias": {
        "value": True
    },
    "model__module__prenormalization": {
        "value": True
    },
    "model__module__kv_compression": {
        "values": [True, False]
    },
    "model__module__kv_compression_sharing": {
        "values": ["headwise", 'key-value']
    },
    "model__module__initialization": {
        "value": "kaiming"
    },
    "model__module__n_layers": {
        "distribution": "q_uniform",
        "min": 1,
        "max": 6
    },
    "model__module__n_heads": {
        "value": 8,
    },
    "model__module__d_ffn_factor": {
        "distribution": "uniform",
        "min": 2. / 3,
        "max": 8. / 3
    },
    "model__module__ffn_dropout": {
        "distribution": "uniform",
        "min": 0,
        "max": 0.5
    },
    "model__module__attention_dropout": {
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
        "max": 1e-3
    },
    "model__optimizer__weight_decay": {
        "distribution": "log_uniform_values",
        "min": 1e-6,
        "max": 1e-3
    },
    "d_token": {
        "distribution": "q_uniform",
        "min": 64,
        "max": 512
    },
    "model__lr_scheduler": {
        "values": [True, False]
    },
}

ft_transformer_config_default = {
    "model__lr_scheduler": {
        "value": False
    },
    "model__module__activation": {
        "value": "reglu"
    },
    "model__module__token_bias": {
        "value": True
    },
    "model__module__prenormalization": {
        "value": True
    },
    "model__module__kv_compression": {
        "value": True
    },
    "model__module__kv_compression_sharing": {
        "value": "headwise"
    },
    "model__module__initialization": {
        "value": "kaiming"
    },
    "model__module__n_layers": {
        "value": 3
    },
    "model__module__n_heads": {
        "value": 8,
    },
    "model__module__d_ffn_factor": {
        "value": 4. / 3
    },
    "model__module__ffn_dropout": {
        "value": 0.1
    },
    "model__module__attention_dropout": {
        "value": 0.2
    },
    "model__module__residual_dropout": {
        "value": 0.0
    },
    "model__lr": {
        "value": 1e-4,
    },
    "model__optimizer__weight_decay": {
        "value": 1e-5,
    },
    "d_token": {
        "value": 192
    },
}

ft_transformer_config_regression = dict(ft_transformer_config,
                                        **skorch_config,
                                        **{
                                            "model_name": {
                                                "value": "ft_transformer_regressor"
                                            },
                                        })

ft_transformer_config_regression_default = dict(ft_transformer_config_default,
                                        **skorch_config_default,
                                        **{
                                            "model_name": {
                                                "value": "ft_transformer_regressor"
                                            },
                                        })

ft_transformer_config_classif = dict(ft_transformer_config,
                                     **skorch_config,
                                     **{
                                         "model_name": {
                                             "value": "ft_transformer"
                                         },
                                     })

ft_transformer_config_classif_default = dict(ft_transformer_config_default,
                                     **skorch_config_default,
                                     **{
                                         "model_name": {
                                             "value": "ft_transformer"
                                         },
                                     })

############################
# Resnet
############################

resnet_config = {
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
resnet_config_default = {
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
}

resnet_config_regression = dict(resnet_config,
                                **skorch_config,
                                **{
                                    "model_name": {
                                        "value": "rtdl_resnet_regressor"
                                    },
                                })

resnet_config_regression_default = dict(resnet_config_default,
                                **skorch_config_default,
                                **{
                                    "model_name": {
                                        "value": "rtdl_resnet_regressor"
                                    },
                                })

resnet_config_classif = dict(resnet_config,
                             **skorch_config,
                             **{
                                 "model_name": {
                                     "value": "rtdl_resnet"
                                 },
                             })

resnet_config_classif_default = dict(resnet_config_default,
                             **skorch_config_default,
                             **{
                                 "model_name": {
                                     "value": "rtdl_resnet"
                                 },
                             })

############################
# MLP
############################

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

############################
# SAINT
############################

saint_config = {
    "model__args__lr": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-3
    },
    "model__args__batch_size": {
        "values": [128, 256],
    },
    "model__args__val_batch_size": {
        "values": [128, 256],
    },
    "model__args__epochs": {
        "value": 300,
    },
    "model__args__early_stopping_rounds": {
        "value": 10,
    },
    "model__args__use_gpu": {
        "value": True,
    },
    "model__args__data_parallel": {
        "value": False,
    },
    "model__args__num_classes": {
        "value": 1,
    },
    "model__args__model_name": {
        "value": 'saint',
    },
    "model_name": {
        "value": "saint",
    },
    "model_type": {
        "value": "tab_survey",
    },
    "model__params__depth": {
        "values": [1, 2, 3, 6, 12],
    },
    "model__params__heads": {
        "values": [2, 4, 8],
    },
    "model__params__dim": {
        "values": [32, 64, 128]
    },
    "model__params__dropout": {
        "values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
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

saint_config_default = {
        "model__args__lr": {
            "value": float(3e-5),
        },
        "model__args__batch_size": {
            "values": [128],
        },
        "model__args__val_batch_size": {
            "values": [128],
        },
        "model__args__epochs": {
            "value": 300,
        },
        "model__args__early_stopping_rounds": {
            "value": 10,
        },
        "model__args__use_gpu": {
            "value": True,
        },
        "model__args__data_parallel": {
            "value": False,
        },
        "model__args__num_classes": {
            "value": 1,
        },
        "model__args__model_name": {
            "value": 'saint',
        },
        "model_name": {
            "value": "saint",
        },
        "model_type": {
            "value": "tab_survey",
        },
        "model__params__depth": {
            "values": [3],
        },
        "model__params__heads": {
            "values": [4],
        },
        "model__params__dim": {
            "values": [128],
        },
        "model__params__dropout": {
            "values": [0.1],
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
        "values": [False]
    },

}

saint_config_regression = dict(saint_config,
                               **{
                                   "model__args__objective": {
                                       "value": "regression",
                                   },
                               })

saint_config_regression_default = dict(saint_config_default,
                               **{
                                   "model__args__objective": {
                                       "value": "regression",
                                   },
                               })

saint_config_classif = dict(saint_config,
                            **{
                                "model__args__objective": {
                                    "value": "binary",
                                },
                            })

saint_config_classif_default = dict(saint_config_default,
                            **{
                                "model__args__objective": {
                                    "value": "binary",
                                },
                            })


config_dic = {
    "gbt": {
        "classif": {"random": gbt_config_classif,
                    "default": gbt_config_classif_default},
        "regression": {"random": gbt_config_regression,
                       "default": gbt_config_regression_default},
    },
    "rf": {
        "classif": {"random": rf_config_classif,
                    "default": rf_config_classif_default},
        "regression": {"random": rf_config_regression,
                          "default": rf_config_regression_default},
    },
    "xgb": {
        "classif": {"random": xgb_config_classif,
                    "default": xgb_config_classif_default},
        "regression": {"random": xgb_config_regression,
                          "default": xgb_config_regression_default},
    },
    "hgbt": {
        "classif": {"random": hgbt_config_classif,
                    "default": hgbt_config_classif_default},
        "regression": {"random": hgbt_config_regression,
                            "default": hgbt_config_regression_default},
    },
    "ft_transformer": {
        "classif": {"random": ft_transformer_config_classif,
                    "default": ft_transformer_config_classif_default},
        "regression": {"random": ft_transformer_config_regression,
                            "default": ft_transformer_config_regression_default},
    },
    "resnet": {
        "classif": {"random": resnet_config_classif,
                    "default": resnet_config_classif_default},
        "regression": {"random": resnet_config_regression,
                            "default": resnet_config_regression_default},
    },
    "mlp": {
        "classif": {"random": mlp_config_classif,
                    "default": mlp_config_classif_default},
        "regression": {"random": mlp_config_regression,
                            "default": mlp_config_regression_default},
    },
    "saint": {
        "classif": {"random": saint_config_classif,
                    "default": saint_config_classif_default},
        "regression": {"random": saint_config_regression,
                            "default": saint_config_regression_default},
    }
}