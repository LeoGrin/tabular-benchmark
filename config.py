from target_function_classif import *
from data_transforms import *
from generate_data import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, \
    RandomForestRegressor
#from pyearth import Earth
from rotation_forest import RotationForestClassifier
from utils.skorch_utils import create_mlp_skorch, create_mlp_ensemble_skorch, create_sparse_model_skorch, create_sparse_model_new_skorch, create_mlp_skorch_regressor
#from autosklearn.classification import AutoSklearnClassifier
from models.MLRNN import MLRNNClassifier
from models.NAM import create_nam_skorch
from xp_regression import start_from_pretrained_model


#WARNING: name "method" and "method_name" ARE RESERVED




def config(keyword):
    keyword = str(keyword)
    if keyword == "1":
        data_generation_functions = [{"method":generate_gaussian_data,
                                      "method_name":"gaussian",
                                     "num_samples": [1000, 10000],
                                     "num_features": [15, 100],
                                     "cov_matrix":["identity", "random", "random_sparse", "random_sparse_precision"]},
                                    {"method":generate_student_data,
                                     "method_name":"student",
                                     "num_samples": [1000, 10000],
                                     "num_features": [15, 100],
                                     "df":[1, 2, 5]}]
        target_generation_functions = [{"method":generate_labels_random_forest,
                                        "method_name":"random_forest",
                                      "n_trees":[5, 20],
                                      "max_depth":[5, 20],
                                      "depth_distribution":["uniform"],
                                      "split_distribution":["uniform"]},
                                     {"method":generate_labels_linear,
                                      "method_name":"linear",
                                      "noise_level":[0.2],
                                      "weights":["equal", "random"]}]
        data_transforms_functions = [[{"method":add_uninformative_features,
                                      "method_name":"add_random_features",
                                      "num_uninformatives":[10, 50]}],
                                     [{"method":apply_random_rotation,
                                      "method_name":"random_rotation"}],
                                     [{"method": add_noise,
                                      "method_name": "add_noise",
                                      "noise_type": ["white"],
                                      "scale": [0.1, 0.5, 1]}]]
    elif keyword == "2":
        data_generation_functions = [{"method": generate_student_data,
                                      "method_name": "student",
                                      "num_samples": [1000, 10000],
                                      "num_features": [15, 100],
                                      "df": [1, 2, 5]}]
        target_generation_functions = [{"method": generate_labels_random_forest,
                                        "method_name": "random_forest",
                                        "n_trees": [5, 20],
                                        "max_depth": [5, 20],
                                        "depth_distribution": ["uniform"],
                                        "split_distribution": ["uniform"]},
                                       {"method": generate_labels_linear,
                                        "method_name": "linear",
                                        "noise_level": [0.2],
                                        "weights": ["equal", "random"]}]
        data_transforms_functions = [[{"method": add_noise,
                                       "method_name": "add_noise",
                                       "noise_type": ["white"],
                                       "scale": [0.1, 0.5]},
                                      {"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["standard", "robust", "quantile", "power", "quantile_uniform"]}]
                                     ]
    elif keyword == "3":
        data_generation_functions = [{"method":import_real_data,
                                      "method_name":"open_ml",
                                      "openml_task_id":[15, 37, 43, 219, 3902, 3903, 3904, 3913, 3917, 9910, 9946, 9952, 9957, 9971, 9976, 9977, 10093, 10101, 167120, 167141]}]
        target_generation_functions = [{"method":None,
                                        "method_name":"no_transform"}]
        data_transforms_functions = [[{"method":add_uninformative_features,
                                      "method_name":"add_random_features",
                                      "num_uninformatives":[10, 50]}],
                                     [{"method":apply_random_rotation,
                                      "method_name":"random_rotation"}],
                                     [{"method": add_noise,
                                      "method_name": "add_noise",
                                      "noise_type": ["white"],
                                      "scale": [0.1, 0.5, 1]}],
                                     [{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["standard", "robust", "quantile", "power", "quantile_uniform"]}]
                                     ]
    elif keyword == "4":
        data_generation_functions = [{"method":import_real_data,
                                      "method_name":"open_ml",
                                      "openml_task_id":[15, 37, 43, 219, 3902, 3903, 3904, 3913, 3917, 9910, 9946, 9952, 9957, 9971, 9976, 9977, 10093, 10101, 167120, 167141],
                                      "max_num_samples": 10000}]
        target_generation_functions = [{"method":None,
                                        "method_name":"no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["standard", "robust", "quantile", "power", "quantile_uniform"]},
                                      {"method": cluster_1d,
                                       "method_name": "quantize",
                                       "n_clusters": [3, 10]}
                                      ]
                                     ]

    elif keyword == "5":
        data_generation_functions = [{"method":import_real_data,
                                      "method_name":"open_ml",
                                      "openml_task_id":[15, 37, 43, 219, 3902, 3903, 3904, 3913, 3917, 9910, 9946, 9952, 9957, 9971, 9976, 9977, 10093, 10101, 167120, 167141],
                                      "max_num_samples": 10000}]
        target_generation_functions = [{"method":None,
                                        "method_name":"no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["standard", "robust", "quantile", "power", "quantile_uniform"]}
                                      ]
                                     ]

    elif keyword == "6":
        data_generation_functions = [{"method":import_real_data,
                                      "method_name":"open_ml",
                                      "openml_task_id":[15, 37, 43, 219, 3902, 3903, 3904, 3913, 3917, 9910, 9946, 9952, 9957, 9971, 9976, 9977, 10093, 10101, 167120, 167141],
                                      "max_num_samples": 10000}]
        target_generation_functions = [{"method":None,
                                        "method_name":"no_transform"}]
        data_transforms_functions = [[{"method": None,
                                      "method_name": "no_transform"}]]
    elif keyword == "7":
        data_generation_functions = [{"method":generate_gaussian_data,
                                      "method_name":"gaussian",
                                     "num_samples": [1000, 10000],
                                     "num_features": [15, 100],
                                     "cov_matrix":["identity", "random", "random_sparse", "random_sparse_precision"]}]
        target_generation_functions = [{"method":last_column_as_target,
                                        "method_name":"last_column"}]
        data_transforms_functions = [[{"method":remove_last_column,
                                       "method_name":"remove_last_column"},
                                      {"method":add_uninformative_features,
                                      "method_name":"add_random_features",
                                      "num_uninformatives":[10, 50]}],
                                     [{"method":remove_last_column,
                                       "method_name":"remove_last_column"}],
                                     [{"method":remove_last_column,
                                       "method_name":"remove_last_column"},
                                      {"method": add_noise,
                                      "method_name": "add_noise",
                                      "noise_type": ["white"],
                                      "scale": [0.1, 0.5, 1]}]]
    elif keyword == "8":
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "open_ml",
                                      "openml_task_id": [15, 37, 43, 219, 3902, 3903, 3904, 3913, 3917, 9910, 9946,
                                                         9952, 9957, 9971, 9976, 9977, 10093, 10101, 167120, 167141],
                                      "max_num_samples": 10000}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": remove_pseudo_categorial,
                                       "method_name": "remove_pseudo_categorial",
                                       "threshold":[10, 20, 0.1, 0.2]}],
                                     [{"method": remove_pseudo_categorial,
                                       "method_name": "remove_pseudo_categorial",
                                       "threshold": [10, 20, 0.1, 0.2]},
                                      {"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["standard", "robust", "quantile", "power", "quantile_uniform"]}]
                                     ]

    elif keyword == "rotation":
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "keyword": ["219", "3904", "california", "covtype", "spam"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method":None, "method_name":"no_transform"}],
                                     [{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["standard", "robust", "quantile"]}],
                                     [{"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["standard", "robust", "quantile"]},
                                      {"method": apply_random_rotation,
                                       "method_name":"random_rotation"}],
                                     [{"method": apply_random_rotation,
                                       "method_name": "random_rotation"}]
                                     ]

    elif keyword == "rotation_test":
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "keyword": ["219", "3904", "california", "covtype", "spam"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": apply_random_rotation,
                                       "method_name":"random_rotation"},
                                      {"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["standard", "robust", "quantile"]}],
                                     [{"method": apply_random_rotation,
                                       "method_name": "random_rotation"},
                                      {"method":None, "method_name":"no_transform"}]
                                     ]

    elif keyword == "rotation_long":
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "keyword": ["219", "3904", "california", "covtype", "spam"],
                                      "max_num_samples": [None]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method":None, "method_name":"no_transform"}],
                                     [{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["standard", "robust", "quantile"]}],
                                     [{"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["standard", "robust", "quantile"]},
                                      {"method": apply_random_rotation,
                                       "method_name":"random_rotation"}],
                                     [{"method": apply_random_rotation,
                                       "method_name": "random_rotation"}]
                                     ]
    elif keyword == "mlp_ensemble":
        model_generation_functions = [{"method": create_mlp_ensemble_skorch, "hidden_size":[64], "method_name":"mlp_ensemble",
                                       "n_mlps":[10, 20], "mlp_size":[5, 7, 10, 20], "train_on_different_batch":[True, False], "batch_size":512}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]
    elif keyword == "tree_taxonomy_real":
        model_generation_functions = [{"method": RandomForestClassifier, "max_depth": [None], "n_estimators":[100], "max_features":["sqrt", "log2", None, 2, 3], "method_name":"rf"}]
           #{"method": RandomForestClassifier, "max_depth": [None], "n_estimators": [5], "method_name": "rf"}]
                                      #{"method": RandomForestClassifier, "max_depth": [None, 3, 5, 10, 20], "n_estimators":[5, 10, 40, 100, 200], "method_name":"rf"},
                  #{"method": HistGradientBoostingClassifier, "max_depth": [None, 3, 5, 10, 20, 30], "method_name":"hgbt"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao"],
                                      "max_num_samples": [10000]}]#, 30000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method":None, "method_name":"no_transform"}]]
                                    # [{"method": apply_random_rotation,
                                  #     "method_name":"random_rotation"}]]
#

    elif keyword == "tree_taxonomy_synth":
        model_generation_functions = [{"method": RandomForestClassifier, "max_depth": [None, 3, 5, 10, 20], "n_estimators":[5, 10, 40, 100, 200], "method_name":"rf"},
                  {"method": HistGradientBoostingClassifier, "max_depth": [None, 3, 5, 10, 20, 30], "method_name":"hgbt"}]
        data_generation_functions = [{"method":generate_gaussian_data,
                                      "method_name":"gaussian",
                                     "num_samples": [1000, 10000],
                                     "num_features": [15, 100],
                                     "cov_matrix":["identity", "random"]}]
        target_generation_functions = [{"method": generate_labels_random_forest,
                                        "method_name": "random_forest",
                                        "n_trees": [5, 20],
                                        "max_depth": [5, 20],
                                        "depth_distribution": ["uniform"],
                                        "split_distribution": ["uniform"]}]
        data_transforms_functions = [[{"method":None, "method_name":"no_transform"}],
                                     [{"method": apply_random_rotation,
                                       "method_name":"random_rotation"}]]

    elif keyword == "new_gaussian":
        model_generation_functions = [{"method": RandomForestClassifier, "max_depth": [None, 3], "method_name":"rf"},
                  {"method": HistGradientBoostingClassifier, "method_name":"hgbt"},
                  {"method": create_mlp_skorch, "hidden_size":[256], "method_name":"mlp", "batch_size":512}]
        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [10000],
                                      "num_features": [15, 100],
                                      "cov_matrix": ["identity", "random", "random_sparse", "random_sparse_precision"]}]
        target_generation_functions = [{"method": last_column_as_target,
                                        "method_name": "last_column"}]
        data_transforms_functions = [[{"method": remove_last_column,
                                       "method_name": "remove_last_column"}]]

    elif keyword == "random_search_cpu":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name": "rf",
                                       "max_depth": [None, 3, 5, 10, 20],
                                       "criterion": ["gini", "entropy"],
                                       "n_estimators": [50, 100, 200],
                                       "min_samples_split": [2, 10, 20],
                                       "min_samples_leaf": [1, 5, 10],
                                       "min_weight_fraction_leaf": [0, 0.1],
                                       "max_features": [3, "sqrt", "log2", None],
                                       "max_leaf_nodes": [None, 10, 20],
                                      # "min_impurity_decrease": [0, 0.1],
                                      # "ccp_alpha": [0, 0.1],
                                       "max_samples": [0.1, 0.5, 1]},
                                      {"method": ExtraTreesClassifier, "method_name": "xtrrf",
                                       "max_depth": [None, 3, 5, 10, 20],
                                       "criterion": ["gini", "entropy"],
                                       "n_estimators": [50, 100, 200],
                                       "min_samples_split": [2, 10, 20],
                                       "min_samples_leaf": [1, 5, 10],
                                       "min_weight_fraction_leaf": [0, 0.1],
                                       "max_features": [3, "sqrt", "log2", None],
                                       "max_leaf_nodes": [None, 10, 20],
                                       #"min_impurity_decrease": [0, 0.1],
                                       #"ccp_alpha": [0, 0.1],
                                       "max_samples": [0.1, 0.5, 1]},
                                      {"method": HistGradientBoostingClassifier, "method_name": "hgbt",
                                       "learning_rate":[0.01, 0.1, 0.5, 1],
                                       "max_iter": [50, 100, 200],
                                       "max_leaf_nodes": [5, 10, 31, None],
                                       "max_depth": [None, 3, 5, 10, 20],
                                       "min_samples_leaf": [10, 20, 50],
                                       "l2_regularization": [0, 0.1, 0.3],
                                       "max_bins": [50, 100, 255],
                                       "early_stopping":[True, False]}, #TODO: tune early stopping params ?
                                      {"method": RotationForestClassifier, "method_name": "rotf",
                                       "n_features_per_subset": [2, 3, 5],
                                        "rotation_algo": ['pca', "randomized"],
                                        "criterion": ["gini", "entropy"],
                                        "max_depth": [None, 3, 5, 10, 20],
                                        "min_samples_split": [2, 10, 20],
                                       "min_samples_leaf": [1, 5, 10],
                                       "min_weight_fraction_leaf": [0, 0.1],
                                       "max_features": [3, "sqrt", "log2", None],
                                       "max_leaf_nodes": [None, 10, 20]}]#,
                                      # {"method": LogisticRegression, "method_name":"log_reg",
                                      #  "penalty": ["l1", "l2", "elasticnet", "none"],
                                      #  "C": [0.1, 0.5, 1, 2, 5],
                                      #  "max_iter": [50, 100, 200],
                                      #  "solver": ["sag", "saga"]}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]

    elif keyword == "random_search_mnist":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name": "rf",
                                       "max_depth": [None, 3, 5, 10, 20],
                                       "criterion": ["gini", "entropy"],
                                       "n_estimators": [50, 100, 200],
                                       "min_samples_split": [2, 10, 20],
                                       "min_samples_leaf": [1, 5, 10],
                                       "min_weight_fraction_leaf": [0, 0.1],
                                       "max_features": [3, "sqrt", "log2", None],
                                       "max_leaf_nodes": [None, 10, 20],
                                       # "min_impurity_decrease": [0, 0.1],
                                       # "ccp_alpha": [0, 0.1],
                                       "max_samples": [0.1, 0.5, 1]},
                                      {"method": HistGradientBoostingClassifier, "method_name": "hgbt",
                                       "learning_rate": [0.01, 0.1, 0.5, 1],
                                       "max_iter": [50, 100, 200],
                                       "max_leaf_nodes": [5, 10, 31, None],
                                       "max_depth": [None, 3, 5, 10, 20],
                                       "min_samples_leaf": [10, 20, 50],
                                       "l2_regularization": [0, 0.1, 0.3],
                                       "max_bins": [50, 100, 255],
                                       "early_stopping": [True, False]}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["mnist_1_7"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["quantile"]},
                                      {"method": None, "method_name": "no_transform"}]]
    elif keyword == "random_search_rot":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name": "rf",
                                       "max_depth": [None, 3, 5, 10, 20],
                                       "criterion": ["gini", "entropy"],
                                       "n_estimators": [50, 100, 200],
                                       "min_samples_split": [2, 10, 20],
                                       "min_samples_leaf": [1, 5, 10],
                                       "min_weight_fraction_leaf": [0, 0.1],
                                       "max_features": [3, "sqrt", "log2", None],
                                       "max_leaf_nodes": [None, 10, 20],
                                      # "min_impurity_decrease": [0, 0.1],
                                      # "ccp_alpha": [0, 0.1],
                                       "max_samples": [0.1, 0.5, 1]},
                                      {"method": HistGradientBoostingClassifier, "method_name": "hgbt",
                                       "learning_rate":[0.01, 0.1, 0.5, 1],
                                       "max_iter": [50, 100, 200],
                                       "max_leaf_nodes": [5, 10, 31, None],
                                       "max_depth": [None, 3, 5, 10, 20],
                                       "min_samples_leaf": [10, 20, 50],
                                       "l2_regularization": [0, 0.1, 0.3],
                                       "max_bins": [50, 100, 255],
                                       "early_stopping":[True, False]}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["california", "churn", "credit", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                     {"method": apply_random_rotation,
                                      "method_name": "random_rotation"}]]


    elif keyword == "mlp_cpu_default":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name": "rf",
                                       "n_features_per_tree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                                       "oob_score": [True],
                                       "oob_score_per_tree": [True, False],
                                       "threshold_estimator_selection": [0.6, 0.8, 0.9],
                                       #"n_estimators": [100],
                                       "max_depth":[None, 3, 5, 10],
                                       "max_features":[None],
                                       "n_estimators":[100, 300]}]
                                     # {"method": ExtraTreesClassifier, "method_name": "xtrrf"},
                                     # {"method": HistGradientBoostingClassifier, "method_name": "hgbt"},
                                     # {"method": RotationForestClassifier, "method_name": "rotf"},
                                     # {"method": LogisticRegression, "method_name":"log_reg"},
                                     #  {"method": create_mlp_skorch,
                                     #     "n_layers": [3],
                                     #     "hidden_size": [128],
                                     #     "module__activations": ["selu", "relu"],
                                     #     "lr": [0.001],
                                     #     "batch_size": [128],
                                     #     "method_name": "mlp",
                                     #     "device": "cpu"}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      #"keyword": ["heloc", "electricity_new"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]
                                    # [{"method": gaussienize,
                                    #  "method_name": "gaussienize",
                                    #  "type": ["quantile"]},
                                    # {"method": apply_random_rotation,
                                    #  "method_name": "random_rotation"}]]

    elif keyword == "rotation_experiment":
        model_generation_functions = [{"method": RandomForestClassifier,
                                       "method_name": "rf"},
                                      {"method": HistGradientBoostingClassifier,
                                       "method_name": "hgbt"},
                                      {"method": create_mlp_skorch,
                                        "optimizer": ["adam"],
                                        #"optimizer__weight_decay": [0.01, 1.0],
                                        "max_epochs": [1000],
                                        "n_layers": [4],
                                        "hidden_size": [256],
                                        "lr": [0.0005],
                                        "batch_size": [512],
                                        "es_patience": [200],
                                        "lr_patience": [30],
                                        "method_name": "mlp",
                                        "device": "cpu"}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "mnist_1_7", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method":remove_features_rf,
                                       "method_name":"features_selection",
                                       "num_features_to_remove": [0, 0.3, 0.5, 0.6]},
                                      {"method": apply_random_rotation,
                                       "method_name": "random_rotation"},
                                      {"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["quantile"]}],
                                     [{"method":remove_features_rf,
                                       "method_name":"features_selection",
                                       "num_features_to_remove": [0, 0.3, 0.5, 0.6]},
                                      {"method": None,
                                      "method_name": "no_transform"},
                                      {"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["quantile"]}]
                                     ]


    elif keyword == "depth_impact":
        model_generation_functions = [{"method": RandomForestClassifier,
                                       "method_name": "rf",
                                       "max_depth":[2, 3, 5, 10, None],
                                       "n_estimators":[10, 50, 100, 200]},
                                      {"method": HistGradientBoostingClassifier,
                                       "method_name": "hgbt",
                                       "learning_rate": [0.01, 0.1, 0.5, 1],
                                       "max_depth": [2, 3, 5, 10, None]}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "mnist_1_7", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}],
                                     [{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                     {"method": apply_random_rotation,
                                      "method_name": "random_rotation"}]]
    elif keyword == "simple_rf":
        model_generation_functions = [{"method": RandomForestClassifier,
                                       "method_name": "rf"}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": None,
                                        "method_name": "no_transform"}]]
    elif keyword == "useless_features":
        model_generation_functions = [{"method": RandomForestClassifier,
                                       "method_name": "rf"},
                                      {"method": HistGradientBoostingClassifier,
                                       "method_name": "hgbt"}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method":remove_features_rf,
                                       "method_name":"features_selection",
                                       "num_features_to_remove": [0, 1, 2, 3, 4]}],
                                     [{"method": remove_features_rf,
                                      "method_name": "features_selection",
                                      "num_features_to_remove": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}],
                                     [{"method": remove_features_rf,
                                       "method_name": "features_selection",
                                       "importance_cutoff": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15]}]
                                     ]


    elif keyword == "random_search_gpu_default":
        model_generation_functions = [{"method": create_nam_skorch,
                                      "method_name":"nam",
                                      "mlp_hidden_sizes":[None, [128], [256], [128, 128]],
                                       "device": "cuda"}]
                                      #{"method": MLRNNClassifier, "method_name": "lounici"}]
                                      #{"method": create_mlp_skorch, "hidden_size":[256, 512], "batch_size":[128], "method_name":"mlp"}]
                  #{"method": create_mlp_ensemble_skorch, "hidden_size":[64, 128], "method_name":"mlp_ensemble",
                  # "n_mlps":[5, 10], "mlp_size":[3, 5], "train_on_different_batch":False, "batch_size":512}]
           #{"method":MLPClassifier, "method_name":"sklearn_mlp", "hidden_layer_sizes":(256, 256), "max_iter":400}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]

    elif keyword == "nam":
        model_generation_functions = [{"method": create_nam_skorch,
                                      "method_name":"nam",
                                      "module__mlp_hidden_sizes":[None, [128], [256], [128, 128]],
                                       "module__embedding_size": [1, 10, 100],
                                       "device": "cuda"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]


    elif keyword == "sparse":
        model_generation_functions = [{"method": create_sparse_model_skorch,
                                      "method_name":"sparse",
                                       "es_patience": [200],
                                       "lr": [0.01],
                                       "lr_patience": [50],
                                       "max_epochs": [250],
                                       "batch_size": [128],
                                       "module__n_w": [5],
                                       "module__x_inside": [True],
                                       "module__n_layers": [5],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0.0],
                                       "module__hidden_size": [None],
                                       "module__activation_on": ["no", "each"],
                                       "module__n_concat": [1, 2],
                                       "device": "cpu"}]
                                      # {"method": create_mlp_skorch,
                                      #  "optimizer": ["adam"],
                                      #  "optimizer__weight_decay": [0.0],
                                      #  "module__batchnorm": [True],
                                      #  "module__dropout_prob": [0.5],
                                      #  "module__resnet": [False],
                                      #  "max_epochs": [500],
                                      #  "n_layers": [2],
                                      #  "hidden_size": [128],
                                      #  "lr": [0.001],
                                      #  "batch_size": [128],
                                      #  "es_patience": [400],
                                      #  "lr_patience": [200],
                                      #  "method_name": "mlp",
                                      #  "device": "cpu"}]

        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [5000],
                                      "num_features": [15],
                                      "cov_matrix": ["identity"]}]
        target_generation_functions = [{"method": generate_labels_random_forest,
                                        "method_name": "random_forest",
                                        "n_trees": [7],
                                        "max_depth": [3],
                                        "depth_distribution": ["constant"],
                                        "split_distribution": ["uniform"]}]
        data_transforms_functions = [[{"method": balance, "method_name": "balance"}]]

    elif keyword == "sparse_real":
        model_generation_functions = [{"method": create_sparse_model_skorch,
                                      "method_name":"sparse",
                                       "es_patience": [100],
                                       "lr": [0.01],
                                       "lr_patience": [50],
                                       "max_epochs": [200],
                                       "batch_size": [256],
                                       "module__n_w": [5, 10],
                                       "module__x_inside": [True],
                                       "module__temperature": [0.1, 1.0, 10, "sqrt"],
                                       "module__n_layers": [5, 10],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0.0],
                                       "module__hidden_size": [None],
                                       "module__activation_on": ["each", "no"],
                                       "module__n_concat": [1, 2, 5, 20],
                                       "device": "cpu"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      #"keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "keyword": ["electricity", "california"],
                                      #"keyword": ["electricity_new"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]

    elif keyword == "sparse_real_all":
        model_generation_functions = [{"method": create_sparse_model_skorch,
                                      "method_name":"sparse",
                                       "es_patience": [100],
                                       "lr": [0.01],
                                       "lr_patience": [50],
                                       "max_epochs": [350],
                                       "batch_size": [256],
                                       "module__n_w": [5],
                                       "module__x_inside": [True],
                                       "module__temperature": [1.0],
                                       "module__n_layers": [5],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0.0],
                                       "module__hidden_size": [None],
                                       "module__activation_on": ["each", "no"],
                                       "module__n_concat": [5],
                                       "device": "cpu"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      #"keyword": ["electricity", "california"],
                                      #"keyword": ["electricity_new"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]


    elif keyword == "sparse_in_interaction":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name": "rf",
                                       "n_features_per_tree":[0.1, 0.5, 0.7],
                                       "max_depth": [5, 10, None],
                                       "max_features":["auto", None]}]
                                      #{"method": RotationForestClassifier, "method_name": "rotf"}]
        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [5000],
                                      "num_features": [30],
                                      "cov_matrix": ["identity", "random_sparse_precision"]}]
        target_generation_functions = [{"method": generate_labels_sparse_in_interaction,
                                        "method_name": "sparse_in_interaction",
                                        "ensemble_size": [0.1, 0.5, 0.7, 1.0],
                                        "n_interactions": [3, 10],
                                        "variant": ["transform_sum"]}]
        data_transforms_functions = [[{"method": None, "method_name": "no_transform"}]]

    elif keyword == "sparse_in_interaction_gpu":
        model_generation_functions = [{"method": create_mlp_skorch,
                                       "optimizer": ["adam"],
                                       "optimizer__weight_decay": [0.0],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0, 0.5],
                                       "module__resnet": [False],
                                       "max_epochs": [200],
                                       "n_layers": [2],
                                       "hidden_size": [128],
                                       "lr": [0.001],
                                       "batch_size": [128],
                                       "es_patience": [100],
                                       "lr_patience": [50],
                                       "method_name": "mlp",
                                       "device": "cuda"}]
                                      #{"method": RotationForestClassifier, "method_name": "rotf"}]
        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [5000],
                                      "num_features": [30],
                                      "cov_matrix": ["identity", "random_sparse_precision"]}]
        target_generation_functions = [{"method": generate_labels_sparse_in_interaction,
                                        "method_name": "sparse_in_interaction",
                                        "ensemble_size": [0.1, 0.3, 0.5, 0.7, 1.0],
                                        "n_interactions": [3, 10],
                                        "variant": ["transform_sum"]}]
        data_transforms_functions = [[{"method": None, "method_name": "no_transform"}]]
    elif keyword == "sparse_in_interaction_streamlit":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name": "rf", "n_features_per_tree":[0.1, 0.5]}]
                                      #{"method": RotationForestClassifier, "method_name": "rotf"}]
        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [5000],
                                      "num_features": [2],
                                      "cov_matrix": ["identity", "random_sparse_precision"]}]
        target_generation_functions = [{"method": generate_labels_sparse_in_interaction,
                                        "method_name": "sparse_in_interaction",
                                        "ensemble_size": [2],
                                        "n_interactions": [1, 3, 5],
                                        "variant": ["sum", "transform_sum", "hierarchical"]}]
        data_transforms_functions = [[{"method": None, "method_name": "no_transform"}]]

    elif keyword == "lounici":
        model_generation_functions = [{"method": MLRNNClassifier,
                                       "method_name": "lounici",
                                       "scheduler":[True],
                                       "depth": [1, 2, 3],
                                       "width": [2048, 4096],
                                       "learning_rate":[1e-2, 1e-3, 1e-4]}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]

    elif keyword == "mlp":
        model_generation_functions = [{"method": create_mlp_skorch,
                                       "optimizer": ["adam"],
                                       "optimizer__weight_decay": [0.0],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0, 0.5],
                                       "module__activations": ["relu"],
                                       "module__resnet": [False],
                                       "max_epochs": [350],
                                       "n_layers": [3],
                                       "hidden_size":[256],
                                       "lr": [0.001],
                                       "batch_size": [512],
                                       "es_patience": [400],
                                       "lr_patience": [50],
                                       "method_name":"mlp",
                                       "device": "cuda"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      #"keyword": ["electricity_new"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]

    elif keyword == "mlp_trees":
        model_generation_functions = [{"method": create_mlp_skorch,
                                       "optimizer": ["adam"],
                                       "optimizer__weight_decay": [0.0],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0, 0.5],
                                       "module__activations": ["relu"],
                                       "module__resnet": [False],
                                       "max_epochs": [500],
                                       "n_layers": [3],
                                       "hidden_size":[256],
                                       "lr": [0.001],
                                       "batch_size": [512],
                                       "es_patience": [65],
                                       "lr_patience": [25],
                                       "method_name":"mlp",
                                       "device": "cuda"}]
        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [5000],
                                      "num_features": [15],
                                      "cov_matrix": ["identity"]}]
        target_generation_functions = [{"method": generate_labels_random_forest,
                                        "method_name": "random_forest",
                                        "n_trees": [7],
                                        "max_depth": [3],
                                        "depth_distribution": ["constant"],
                                        "split_distribution": ["uniform"]}]
        data_transforms_functions = [[{"method": balance, "method_name": "balance"}]]

    elif keyword == "sparse_new_test":
        model_generation_functions = [{"method": create_sparse_model_new_skorch,
                                       "method_name": "sparse_new",
                                       "batch_size": [256],
                                       "lr": [0.001],
                                       "module__linear_output_layer": [False],
                                       "module__temperature": [0.2],
                                       "module__n_layers": [5],
                                       "module__n_hidden": [1024],
                                       "module__batchnorm": [True],
                                       "max_epochs": [200]}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["covtype"],
                                      # "keyword": ["electricity_new"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["quantile"]},
                                      {"method": None, "method_name": "no_transform"}]]

    elif keyword == "sparse_new":
        model_generation_functions = [{"method": create_sparse_model_new_skorch,
                                       "method_name": "sparse_new",
                                       "batch_size": [256],
                                       "lr": [0.01],
                                       "lr_patience": [10],
                                       "es_patience": [25],
                                       "module__x_inside": [False],
                                       "module__bias": [True],
                                       "module__train_temperature": [False],
                                       "module__concatenate_input": [True],
                                       "module__linear_output_layer": [False],
                                       "module__dropout_prob": [0.2],
                                       "module__temperature": [0.1, 0.5, 1.0],
                                       "module__train_selectors": [True],
                                       "module__n_layers": [2, 5],
                                       "module__n_hidden": [1048],
                                       "module__batchnorm": [True],
                                       "max_epochs": [450],
                                       "device": "cuda"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      #"keyword": ["credit", "shopping", "wine", "nomao", "cpu"],
                                      #"keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn"],
                                      "keyword": ["covtype", "electricity", "california", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["quantile"]},
                                      {"method": None, "method_name": "no_transform"}]]


    elif keyword == "sparse_new_no_train":
        model_generation_functions = [{"method": create_sparse_model_new_skorch,
                                       "method_name": "sparse_new",
                                       "batch_size": [256],
                                       "lr": [0.01],
                                       "lr_patience": [20],
                                       "es_patience": [50],
                                       "module__x_inside": [False],
                                       "module__bias": [True],
                                       "module__train_temperature": [False],
                                       "module__concatenate_input": [True],
                                       "module__linear_output_layer": [False],
                                       "module__dropout_prob": [None],
                                       "module__temperature": [0.1, 0.5, 1.0],
                                       "module__train_selectors": [False],
                                       "module__n_layers": [2, 5],
                                       "module__n_hidden": [5092],
                                       "module__batchnorm": [True],
                                       "max_epochs": [450],
                                       "device": "cuda"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      #"keyword": ["credit", "shopping", "wine", "nomao", "cpu"],
                                      #"keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn"],
                                      "keyword": ["covtype", "electricity", "california", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                       "method_name": "gaussienize",
                                       "type": ["quantile"]},
                                      {"method": None, "method_name": "no_transform"}]]


    elif keyword == "sparse_new_trees":
        model_generation_functions = [{"method": create_sparse_model_new_skorch,
                                       "method_name": "sparse_new",
                                       "batch_size": [256],
                                       "lr": [0.01],
                                       "lr_patience": [10],
                                       "es_patience": [25],
                                       "module__x_inside": [False],
                                       "module__bias": [True],
                                       "module__train_temperature": [False],
                                       "module__concatenate_input": [True],
                                       "module__linear_output_layer": [False],
                                       "module__dropout_prob": [None],
                                       "module__temperature": [0.1, 1.0],
                                       "module__train_selectors": [True],
                                       "module__n_layers": [2, 5],
                                       "module__n_hidden": [1048],
                                       "module__batchnorm": [True],
                                       "max_epochs": [450],
                                       "device": "cuda"}]
        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [5000],
                                      "num_features": [15],
                                      "cov_matrix": ["identity"]}]
        target_generation_functions = [{"method": generate_labels_random_forest,
                                        "method_name": "random_forest",
                                        "n_trees": [7],
                                        "max_depth": [3],
                                        "depth_distribution": ["constant"],
                                        "split_distribution": ["uniform"]}]
        data_transforms_functions = [[{"method": balance, "method_name": "balance"}]]

    elif keyword == "mlp_simple":
        model_generation_functions = [{"method": create_mlp_skorch,
                                       "optimizer": ["adamw", "adam"],
                                       "optimizer__weight_decay":[0.01, 1.0],
                                       "max_epochs": [1000],
                                       "n_layers": [4],
                                       "hidden_size":[128],
                                       "lr": [0.001],
                                       "batch_size": [512],
                                       "es_patience": [300],
                                       "lr_patience": [50],
                                       "method_name":"mlp",
                                       "device": "cpu"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]

    elif keyword == "electricity":
        model_generation_functions = [#{"method": RandomForestClassifier, "method_name": "rf"},
                                      {"method": create_mlp_skorch,
                                       "optimizer": ["adam"],
                                       #"update_only_if_improve_val": [True],
                                       #"optimizer__weight_decay": [0],
                                       "module__use_exu": [False],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0],
                                       #"module__feature_dropout_prob": [0.],
                                       "module__activations": ["relu"],
                                       "module__resnet": [True],
                                       "max_epochs": [5000],
                                       "n_layers": [4],
                                       "hidden_size":[1024],
                                       "lr": [0.0001],
                                       "batch_size": [512],
                                       "es_patience": [500],
                                       "lr_patience": [20],
                                       "method_name":"mlp",
                                       "device": "cuda"}
                                      # {"method": create_sparse_model_new_skorch,
                                      #  "method_name": "sparse_new",
                                      #  "batch_size": [256],
                                      #  "lr": [0.01],
                                      #  "lr_patience": [10],
                                      #  "es_patience": [25],
                                      #  "module__x_inside": [False],
                                      #  "module__bias": [True],
                                      #  "module__train_temperature": [False],
                                      #  "module__concatenate_input": [True],
                                      #  "module__linear_output_layer": [False],
                                      #  "module__dropout_prob": [None],
                                      #  "module__temperature": [0.1],
                                      #  "module__train_selectors": [True],
                                      #  "module__n_layers": [2],
                                      #  "module__n_hidden": [1048],
                                      #  "module__batchnorm": [True],
                                      #  "max_epochs": [450],
                                      #  "device": "cpu"}
                                      ]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["electricity", "california"],
                                      #"keyword": ["electricity", "california", "wine", "covtype"],
                                      "max_num_samples": [10000]}] #TODO
        target_generation_functions = [{"method": select_features_rf,
                                        "method_name": "select_features"}]
        data_transforms_functions = [[{"method": select_features_rf,
                                      "method_name":"select_features"},
                                      {"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]}],
                                     [{"method": select_features_rf,
                                       "method_name": "select_features"},
                                      {"method": tree_quantile_transformer,
                                      "method_name": "tree_quantile_transformer"}]]
                                      #{"method":lambda x,y,rng:select_features_rf(x, y, rng, 1),
                                      # "method_name":"select_features"}]]

    elif keyword == "regression":
        model_generation_functions = [{"method": RandomForestRegressor, "method_name": "rf"},
                                      {"method": create_mlp_skorch_regressor,
                                       "optimizer": ["adam"],
                                       "optimizer__weight_decay": [0],
                                       "module__use_exu": [False],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0],
                                       "module__activations": ["relu"],
                                       "module__resnet": [True],
                                       "max_epochs": [150],
                                       "n_layers": [3],
                                       "hidden_size":[256],
                                       "lr": [0.001],
                                       "batch_size": [512],
                                       "es_patience": [110],
                                       "lr_patience": [40],
                                       "method_name":"mlp",
                                       "device": "cpu"}
                                      ]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": False,
                                      "regression": True,
                                      "dim": [["Longitude"]],
                                      #"keyword": ["electricity"],
                                      "keyword": ["california"],
                                      "max_num_samples": [10000]}] #TODO
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]}]]
                                      #{"method":lambda x,y,rng:select_features_rf(x, y, rng, 1),
                                      # "method_name":"select_features"}]]


    elif keyword == "test_gpu":
        model_generation_functions = [#{"method": RandomForestRegressor, "method_name": "rf"},
                                      {"method": create_mlp_skorch_regressor,
                                       "optimizer": ["adam"],
                                       "optimizer__weight_decay": [0],
                                       "module__use_exu": [False],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0],
                                       "module__activations": ["relu"],
                                       "module__resnet": [True],
                                       "max_epochs": [100],
                                       "n_layers": [3],#, 9],
                                       "hidden_size":[248],#, 1024],
                                       "lr": [0.0001],
                                       "batch_size": [256],#[0.1, 0.2, 0.5, 0.95, 256, 1024],
                                       "es_patience": [50],
                                       "lr_patience": [10],
                                       "method_name":"mlp",
                                       "device": "cuda"}
                                      ]
        # data_generation_functions = [{"method": generate_uniform_data,
        #                               "method_name": "uniform",
        #                               "num_samples": [500, 600, 750, 1000, 1600, 2500, 5000],
        #                               "num_features": 1,
        #                               "regression": True}] #TODO
        # target_generation_functions = [{"method": periodic_triangle,
        #                                 "method_name": "periodic",
        #                                 "period": [8],
        #                                 "period_size": [0.1, 0.15, 0.2, 0.3, 0.4],#[0.1, 0.15],#, 0.2, 0.3, 0.4],#, 0.2, 0.3],
        #                                 #"offset": [0, 0.3, 0.5, 0.7, 1, 1.5],
        #                                 "noise": [False]}]
        data_generation_functions = [{"method": generate_periodic_triangles_uniform,
                                      "method_name": "triangle_uniform",
                                      "num_samples": [1000],
                                      "period": [8],
                                      "period_size": [0.2],
                                      "noise": [False],
                                      "regression": True}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_method"}]
        data_transforms_functions = [#[{"method": tree_quantile_transformer,
                                     # "method_name": "tree_quantile_transformer",
                                     #  "regression":True}],
                                     [{"method": None,
                                        "method_name": "no_method",
                                        "regression":True}]
                                     ]

    elif keyword == "regression_synthetic":
        model_generation_functions = [#{"method": RandomForestRegressor, "method_name": "rf"},
                                      {"method": create_mlp_skorch_regressor,
                                       "optimizer": ["adam"],
                                       "use_checkpoints": [False],
                                       "optimizer__weight_decay": [0],
                                       "module__use_exu": [False],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0],
                                       "module__activations": ["relu"],
                                       "module__resnet": [True],
                                       "max_epochs": [1200],
                                       "n_layers": [1, 3],
                                       "hidden_size":[256],
                                       "lr": [0.0001],
                                       "batch_size": [0.1, 0.2, 0.5, 0.95],
                                       "es_patience": [50],
                                       "lr_patience": [10],
                                       "method_name":"mlp",
                                       "device": "cpu"}
                                      ]
        data_generation_functions = [{"method": "uniform_data",
                                      "method_name": "uniform",
                                      "num_samples": [500, 600, 750, 1000, 1600, 2500, 5000, 7500, 10000],
                                      "num_features": 1,
                                      "regression": True}] #TODO
        target_generation_functions = [{"method": "periodic_triangle",
                                        "method_name": "periodic",
                                        "period": [8],
                                        "period_size": [0.1, 0.15, 0.2, 0.3, 0.4],#[0.1, 0.15],#, 0.2, 0.3, 0.4],#, 0.2, 0.3],
                                        #"offset": [0, 0.3, 0.5, 0.7, 1, 1.5],
                                        "noise": [False]}]
        # data_generation_functions = [{"method": generate_periodic_triangles_uniform,
        #                               "method_name": "triangle_uniform",
        #                               "num_samples": [500],#[7000, 10000],
        #                               "period": [8],
        #                               "period_size": [0.1, 0.15, 0.2, 0.3, 0.4],
        #                               "noise": [False],
        #                               "regression": True}]
        # target_generation_functions = [{"method": None,
        #                                 "method_name": "no_method"}]
        data_transforms_functions = [#[{"method": tree_quantile_transformer,
                                     # "method_name": "tree_quantile_transformer",
                                     #  "regression":True}],
                                     [{"method": None,
                                        "method_name": "no_method",
                                        "regression":True}]
                                     ]

    elif keyword == "test_cuda":
        model_generation_functions = [#{"method": RandomForestRegressor, "method_name": "rf"},
                                      {"method": create_mlp_skorch_regressor,
                                       "optimizer": ["adam"],
                                       "optimizer__weight_decay": [0],
                                       "module__use_exu": [False],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0],
                                       "module__activations": ["relu"],
                                       "module__resnet": [True],
                                       "max_epochs": [10],
                                       "n_layers": [3],
                                       "hidden_size":[248],
                                       "lr": [0.0001],
                                       "batch_size": [256],#[0.1, 0.2, 0.5, 0.95, 256, 1024],
                                       "es_patience": [50],
                                       "lr_patience": [10],
                                       "method_name":"mlp",
                                       "device": "cuda"}
                                      ]
        data_generation_functions = [{"method": generate_uniform_data,
                                      "method_name": "uniform",
                                      "num_samples": [500, 600],
                                      "num_features": 1,
                                      "regression": True}] #TODO
        target_generation_functions = [{"method": periodic_triangle,
                                        "method_name": "periodic",
                                        "period": [8],
                                        "period_size": [0.3, 0.4],#[0.1, 0.15, 0.2, 0.3, 0.4],#[0.1, 0.15],#, 0.2, 0.3, 0.4],#, 0.2, 0.3],
                                        #"offset": [0, 0.3, 0.5, 0.7, 1, 1.5],
                                        "noise": [False]}]
        data_transforms_functions = [#[{"method": tree_quantile_transformer,
                                     # "method_name": "tree_quantile_transformer",
                                     #  "regression":True}],
                                     [{"method": None,
                                        "method_name": "no_method",
                                        "regression":True}]
                                     ]

    elif keyword == "regression_synthetic_pretrained":
        model_generation_functions = [#{"method": RandomForestRegressor, "method_name": "rf"},
                                      {"method": start_from_pretrained_model,
                                       "method_name": "pretrained",
                                       "pretrained_model_filename": "saved_models/regression_synthetic/mlp/-6905694080875447095",
                                       "noise_std": [0, 0.01, 0.05],#[0, 0.001,  0.005, 0.01, 0.05, 0.1],
                                       "optimizer": ["adam"],
                                       "optimizer__weight_decay": [0],
                                       "module__use_exu": [False],
                                       "module__batchnorm": [True],
                                       "module__dropout_prob": [0],
                                       "module__activations": ["relu"],
                                       "module__resnet": [True],
                                       "max_epochs": [500],
                                       "n_layers": [3],
                                       "hidden_size": [248],
                                       "lr": [0.0001],
                                       "batch_size": [256],
                                       "es_patience": [110],
                                       "lr_patience": [10],
                                       #"method_name": "mlp",
                                       "device": "cpu"}
                                      ]
        data_generation_functions = [{"method": generate_uniform_data,
                                      "method_name": "uniform",
                                      "num_samples": [500],
                                      "num_features": 1,
                                      "regression": True}]  # TODO
        target_generation_functions = [{"method": periodic_triangle,
                                        "method_name": "periodic",
                                        "period": [8],
                                        "period_size": [0.15],#[0.1, 0.15, 0.2, 0.25, 0.5],  # , 0.2, 0.3],
                                        # "offset": [0, 0.3, 0.5, 0.7, 1, 1.5],
                                        "noise": [False]}]
        data_transforms_functions = [
                                     [{"method": None,
                                       "method_name": "no_method",
                                       "regression": True}]
                                     ]

                                    #[[{"method": None,
                                    #  "method_name": "no_method",
                                    #   "regression":True}]]


                                      #{"method":lambda x,y,rng:select_features_rf(x, y, rng, 1),
                                      # "method_name":"select_features"}]]

    elif keyword == "forest_trees":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name": "rf"},
                                      {"method": HistGradientBoostingClassifier, "method_name": "hgbt"}]
        data_generation_functions = [{"method": generate_gaussian_data,
                                      "method_name": "gaussian",
                                      "num_samples": [5000],
                                      "num_features": [15],
                                      "cov_matrix": ["identity"]}]
        target_generation_functions = [{"method": generate_labels_random_forest,
                                        "method_name": "random_forest",
                                        "n_trees": [2],
                                        "max_depth": [5],
                                        "depth_distribution": ["uniform"],
                                        "split_distribution": ["uniform"]}]
        data_transforms_functions = [[{"method": balance, "method_name": "balance"}]]

    elif keyword == "resnet":
        model_generation_functions = [{"method": create_mlp_skorch,
                                       "n_layers": [1, 2, 3],
                                       "hidden_size":[256, 512],
                                       "lr": [0.01],
                                       "module__resnet": [True],
                                       "batch_size":[128],
                                       "method_name":"mlp",
                                       "device": "cuda"}]
        data_generation_functions = [{"method": import_real_data,
                                      "method_name": "real_data",
                                      "balanced": True,
                                      "keyword": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"],
                                      "max_num_samples": [10000]}]
        target_generation_functions = [{"method": None,
                                        "method_name": "no_transform"}]
        data_transforms_functions = [[{"method": gaussienize,
                                      "method_name": "gaussienize",
                                      "type": ["quantile"]},
                                      {"method":None, "method_name":"no_transform"}]]


    elif keyword == "test":
        model_generation_functions = [{"method": RandomForestClassifier, "method_name":"rf"},
                  {"method": HistGradientBoostingClassifier, "method_name":"hgbt"}]
        data_generation_functions = [{"method":generate_gaussian_data,
                                      "method_name":"gaussian",
                                     "num_samples": [100, 200],
                                     "num_features": [15],
                                     "cov_matrix":["identity"]}]
        target_generation_functions = [{"method":generate_labels_random_forest,
                                        "method_name":"random_forest",
                                      "n_trees":[2],
                                      "max_depth":[5],
                                      "depth_distribution":["uniform"],
                                      "split_distribution":["uniform"]}]
        data_transforms_functions = [[{"method": add_noise,
                                      "method_name": "add_noise",
                                      "noise_type": ["white"],
                                      "scale": [0.1]},
                                      {"method":cluster_1d,
                                       "method_name": "quantize",
                                       "n_clusters":3}]]
    else:
        raise ValueError("Keyword not recognized")

    return model_generation_functions, data_generation_functions, target_generation_functions, data_transforms_functions


def merge_configs(keyword_list):
    data_generation_functions_list, target_generation_functions_list, data_transforms_functions_list = [], [], []
    for keyword in keyword_list:
        model_generation_functions, data_generation_functions, target_generation_functions, data_transforms_functions = config(keyword)
        data_generation_functions_list.extend(data_generation_functions)
        target_generation_functions_list.extend(target_generation_functions)
        data_transforms_functions_list.extend(data_transforms_functions)
    return data_generation_functions_list, target_generation_functions_list, data_transforms_functions_list