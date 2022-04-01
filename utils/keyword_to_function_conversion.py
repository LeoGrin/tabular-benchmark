from target_function_classif import *
from generate_data import *
from utils.skorch_utils import create_mlp_skorch, create_mlp_ensemble_skorch, create_sparse_model_skorch, create_sparse_model_new_skorch, create_mlp_skorch_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def convert_keyword_to_function(keyword):
    if keyword == "rf_c":
        return RandomForestClassifier
    if keyword == "rf_r":
        return RandomForestRegressor
    if keyword == 'mlp_skorch_regressor':
        return create_mlp_skorch_regressor
    elif keyword == "uniform_data":
        return generate_uniform_data
    elif keyword == "periodic_triangle":
        return periodic_triangle
    else:
        raise ValueError("Unknown keyword")