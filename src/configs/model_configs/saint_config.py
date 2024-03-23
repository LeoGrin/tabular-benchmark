
config_random  = {
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
        #"values": [False, True]
        "values": [False]
    },
    "use_gpu": {
        "value": True
    }
}

config_default = {
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
    "use_gpu": {
        "value": True
    }

}

config_regression = dict(config_random ,
                               **{
                                   "model__args__objective": {
                                       "value": "regression",
                                   },
                               })

config_regression_default = dict(config_default,
                               **{
                                   "model__args__objective": {
                                       "value": "regression",
                                   },
                               })

config_classif = dict(config_random ,
                            **{
                                "model__args__objective": {
                                    "value": "binary",
                                },
                                "model__args__early_stop_on": {
                                  "value": "valid_acc",
                              },
                            })

config_classif_default = dict(config_default,
                            **{
                                "model__args__objective": {
                                    "value": "binary",
                                },
                                "model__args__early_stop_on": {
                                  "value": "valid_acc",
                              },
                            })
