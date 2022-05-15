from target_function_classif import *
from generate_data import *
from data_transforms import *
from utils.skorch_utils import create_mlp_skorch_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from skorch_models import create_resnet_skorch, create_ft_transformer_skorch, create_rtdl_mlp_skorch, create_NPT_skorch
from skorch_models_regression import create_resnet_regressor_skorch, create_ft_transformer_regressor_skorch
from rotation_forest import RotationForestClassifier


def convert_keyword_to_function(keyword):
    print(keyword)
    if keyword == "npt":
        return create_NPT_skorch
    if keyword == "rotation_forest":
        return RotationForestClassifier
    if keyword == "rtdl_mlp":
        return create_rtdl_mlp_skorch
    if keyword == "ft_transformer":
        return create_ft_transformer_skorch
    if keyword == "ft_transformer_regressor":
        return create_ft_transformer_regressor_skorch
    if keyword == "rtdl_resnet":
        return create_resnet_skorch
    if keyword == "rtdl_resnet_regressor":
        return create_resnet_regressor_skorch
    if keyword == "rf_c":
        return RandomForestClassifier
    if keyword == "rf_r":
        return RandomForestRegressor
    if keyword == "gbt_c":
        return GradientBoostingClassifier
    if keyword == "gbt_r":
        return GradientBoostingRegressor
    if keyword == "xgb_c":
        return XGBClassifier
    if keyword == 'mlp_skorch_regressor':
        return create_mlp_skorch_regressor
    elif keyword == "uniform_data":
        return generate_uniform_data
    elif keyword == "periodic_triangle":
        return periodic_triangle
    elif keyword == "real_data":
        return import_real_data
    elif keyword == "gaussienize":
        return gaussienize
    elif keyword == "select_features_rf":
        return select_features_rf
    elif keyword == "remove_features_rf":
        return remove_features_rf
    elif keyword == "remove_useless_features":
        return remove_useless_features
    elif keyword == "add_uninformative_features":
        return add_uninformative_features
    elif keyword == "random_rotation":
        return apply_random_rotation
    elif keyword == "remove_high_frequency_from_train":
        return remove_high_frequency_from_train
    elif keyword == "no_transform":
        return None
    else:
        raise ValueError("Unknown keyword")