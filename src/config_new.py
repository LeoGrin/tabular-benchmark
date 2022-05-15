import scipy.stats.distributions as distrib

class q_log_uniform(object):
    def __init__(self, low=0, high=1, rng=None):
        self.low = low
        self.high = high
        self.rng = rng
    def rvs(self, size=1):
        distrib.loguniform(self.low, self.high, rng=self.rng).rvs(size)



#WARNING: name "method" and "method_name" ARE RESERVED


def config(keyword):
    keyword = str(keyword)

    if keyword == "simple_rf":
        general_parameters = {"n_iter": "auto",
                              "train_prop": 0.70,
                              "max_train_samples": 10000,
                              "val_test_prop": 0.3,
                              "max_val_samples": 50000,
                              "max_test_samples": 50000,
                              "regression": False}
        model_generation_functions = [{"method_name": "rf_c"}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method_name": "no_transform"}]
        data_transforms_functions = [[{"method_name": "remove_high_frequency"}]]

    elif keyword == "resnet_sweep":
        general_parameters = {"n_iter": "auto",
                              "train_prop": 0.70,
                              "max_train_samples": 10000,
                              "val_test_prop": 0.3,
                              "max_val_samples": 50000,
                             "max_test_samples": 50000,
                              "regression": False}
        model_generation_functions = [{"method_name": "rtdl_resnet",
                                       #"log_training": True,
                                       "device": "cuda",
                                       "optimizer": "adamw",
                                       "lr_scheduler": [True, False],
                                       "batch_size": [256, 512, 1024],
                                       "max_epochs": 300,
                                       "module__activation": "reglu",
                                       "module__normalization": ["batchnorm", "layernorm"],
                                       "module__n_layers": distrib.randint(1, 16),
                                       "module__d": distrib.randint(64, 1024),
                                       "module__d_hidden_factor": distrib.uniform(1, 4),
                                       "module__hidden_dropout": distrib.uniform(0, 0.5),
                                       "module__residual_dropout": distrib.uniform(0, 0.5),
                                       "lr": distrib.loguniform(1e-5, 1e-2),
                                       "optimizer__weight_decay": distrib.loguniform(1e-8, 1e-3),
                                       "module__d_embedding": distrib.randint(64, 512),
                                       }]
        data_generation_functions = [{"method_name": "real_data",
                                      "keyword": ["electricity",
                                                 "covertype",
                                                 "poker",
                                                 "pol",
                                                 "house_16H",
                                                 "kdd_ipums_la_97-small",
                                                 "MagicTelescope",
                                                 "bank-marketing",
                                                 "phoneme",
                                                 "MiniBooNE",
                                                 "Higgs",
                                                 "eye_movements",
                                                 "jannis",
                                                 "credit",
                                                 "california",
                                                 "wine"]
                                      }]
        target_generation_functions = [{"method_name": "no_transform"}]
        data_transforms_functions = [[{"method_name": "gaussienize",
                                       "type": "quantile"}]]
    elif keyword == "resnet_add_features":
        general_parameters = {"n_iter": "auto",
                              "train_prop": 0.70,
                              "max_train_samples": 10000,
                              "val_test_prop": 0.3,
                              "max_val_samples": 50000,
                             "max_test_samples": 50000,
                              "regression": False}
        model_generation_functions = [{"method_name": "rtdl_resnet",
                                       "log_training": True,
                                       "device": "cuda",
                                       "optimizer": "adamw",
                                       "lr_scheduler": [True, False],
                                       "batch_size": [256, 512, 1024],
                                       "max_epochs": 300,
                                       "module__activation": "reglu",
                                       "module__normalization": ["batchnorm", "layernorm"],
                                       "module__n_layers": distrib.randint(1, 16),
                                       "module__d": distrib.randint(64, 1024),
                                       "module__d_hidden_factor": distrib.uniform(1, 4),
                                       "module__hidden_dropout": distrib.uniform(0, 0.5),
                                       "module__residual_dropout": distrib.uniform(0, 0.5),
                                       "lr": distrib.loguniform(1e-5, 1e-2),
                                       "optimizer__weight_decay": distrib.loguniform(1e-8, 1e-3),
                                       "module__d_embedding": distrib.randint(64, 512),
                                       }]
        data_generation_functions = [{"method_name": "real_data",
                                      "keyword": ["electricity",
                                                 "covertype",
                                                 "poker",
                                                 "pol",
                                                 "house_16H",
                                                 "kdd_ipums_la_97-small",
                                                 "MagicTelescope",
                                                 "bank-marketing",
                                                 "phoneme",
                                                 "MiniBooNE",
                                                 "Higgs",
                                                 "eye_movements",
                                                 "jannis",
                                                 "credit",
                                                 "california",
                                                 "wine"]
                                      }]
        target_generation_functions = [{"method_name": "no_transform"}]
        data_transforms_functions = [[{"method_name": "add_uninformative_features",
                                       "multiplier": [1., 1.5, 2]},
                                      {"method_name": "gaussienize",
                                       "type": "quantile"}]]

    else:
        raise ValueError("Keyword not recognized")

    return general_parameters, model_generation_functions, data_generation_functions, target_generation_functions, data_transforms_functions


def merge_configs(keyword_list):
    data_generation_functions_list, target_generation_functions_list, data_transforms_functions_list = [], [], []
    for keyword in keyword_list:
        model_generation_functions, data_generation_functions, target_generation_functions, data_transforms_functions = config(keyword)
        data_generation_functions_list.extend(data_generation_functions)
        target_generation_functions_list.extend(target_generation_functions)
        data_transforms_functions_list.extend(data_transforms_functions)
    return data_generation_functions_list, target_generation_functions_list, data_transforms_functions_list