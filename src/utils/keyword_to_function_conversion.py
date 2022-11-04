from target_function_classif import *
from generate_data import *
from data_transforms import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from skorch_models import create_resnet_skorch, create_ft_transformer_skorch, create_rtdl_mlp_skorch#, create_NPT_skorch
from skorch_models_regression import create_resnet_regressor_skorch, create_ft_transformer_regressor_skorch, create_rtdl_mlp_regressor_skorch
from TabSurvey.models.saint import SAINT
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier


convert_keyword_to_function = {
    "tab_pfn": TabPFNClassifier,
    "rtdl_mlp": create_rtdl_mlp_skorch,
    "rtdl_mlp_regressor": create_rtdl_mlp_regressor_skorch,
    "ft_transformer": create_ft_transformer_skorch,
    "ft_transformer_regressor": create_ft_transformer_regressor_skorch,
    "rtdl_resnet": create_resnet_skorch,
    "rtdl_resnet_regressor": create_resnet_regressor_skorch,
    "rf_c": RandomForestClassifier,
    "rf_r": RandomForestRegressor,
    "gbt_c": GradientBoostingClassifier,
    "gbt_r": GradientBoostingRegressor,
    "hgbt_r": HistGradientBoostingRegressor,
    "hgbt_c": HistGradientBoostingClassifier,
    "xgb_c": XGBClassifier,
    "xgb_r": XGBRegressor,
    "saint": SAINT,
    "uniform_data": generate_uniform_data,
    "periodic_triangle": periodic_triangle,
    "real_data": import_real_data,
    "openml": import_open_ml_data,
    "gaussienize": gaussienize,
    "select_features_rf": select_features_rf,
    "remove_features_rf": remove_features_rf,
    "remove_useless_features": remove_useless_features,
    "add_uninformative_features": add_uninformative_features,
    "random_rotation": apply_random_rotation,
    "remove_high_frequency_from_train": remove_high_frequency_from_train,
    "no_transform": None
}


# Prevent circular imports
#TODO: Find a better way to do this
from sklearn.ensemble import StackingClassifier
def create_stacking_classifier(base_estimator_keyword, final_estimator_keyword):
    base_estimator = convert_keyword_to_function[base_estimator_keyword]
    final_estimator = convert_keyword_to_function[final_estimator_keyword]
    return StackingClassifier(estimators=[(base_estimator_keyword, base_estimator())], final_estimator=final_estimator())

convert_keyword_to_function["stacking"] = create_stacking_classifier