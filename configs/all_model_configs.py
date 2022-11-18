from model_configs import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from skorch_models import create_resnet_skorch, create_ft_transformer_skorch, create_rtdl_mlp_skorch#, create_NPT_skorch
from skorch_models_regression import create_resnet_regressor_skorch, create_ft_transformer_regressor_skorch, create_rtdl_mlp_regressor_skorch
from TabSurvey.models.saint import SAINT
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn.linear_model import LogisticRegression


keyword_to_function = {
                "tab_pfn": {"function": TabPFNClassifier,
                            "config":
                "rtdl_mlp": create_rtdl_mlp_skorch,
                "rtdl_mlp_regressor": create_rtdl_mlp_regressor_skorch,
                "ft_transformer": create_ft_transformer_skorch,
                "ft_transformer_regressor": create_ft_transformer_regressor_skorch,
                "rtdl_resnet": create_resnet_skorch,
                "rtdl_resnet_regressor": create_resnet_regressor_skorch,
                "log_reg": LogisticRegression,
                "rf_c": RandomForestClassifier,
                "rf_r": RandomForestRegressor,
                "gbt_c": GradientBoostingClassifier,
                "gbt_r": GradientBoostingRegressor,
                "hgbt_r": HistGradientBoostingRegressor,
                "hgbt_c": HistGradientBoostingClassifier,
                "xgb_c": XGBClassifier,
                "xgb_r": XGBRegressor,
                "saint": SAINT}