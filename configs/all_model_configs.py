from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from skorch_models import create_resnet_skorch, create_ft_transformer_skorch, create_rtdl_mlp_skorch
from skorch_models_regression import create_resnet_regressor_skorch, create_ft_transformer_regressor_skorch, create_rtdl_mlp_regressor_skorch
from TabSurvey.models.saint import SAINT


total_config = {}
model_keyword_dic = {}

## ADD YOU MODEL HERE ##
# from configs.model_configs.your_file import * #replace template.py by your parameters
# keyword = "your_model"
# total_config[keyword] = {
#         "classif": {"random": config_classif,
#                     "default": config_classif_default},
#         "regression": {"random": config_regression,
#                             "default": config_regression_default},
# }
# #these constructor should create an object
# # with fit and predict methods
# model_keyword_dic[config_regression["model_name"]] = YourModelClassRegressor
# model_keyword_dic[config_classif["model_name"]] = YourModelClassClassifier
#############################


from configs.model_configs.gpt_config import *
keyword = "gpt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]] = GradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]] = GradientBoostingClassifier


from configs.model_configs.rf_config import *
keyword = "rf"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = RandomForestRegressor
model_keyword_dic[config_classif["model_name"]] = RandomForestClassifier

from configs.model_configs.hgbt_config import *
keyword = "hgbt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = HistGradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]] = HistGradientBoostingClassifier

from configs.model_configs.xgb_config import *
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]] = XGBClassifier

from configs.model_configs.xgb_config import *
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]] = XGBClassifier

from configs.model_configs.mlp_config import *
keyword = "mlp"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = create_rtdl_mlp_regressor_skorch
model_keyword_dic[config_classif["model_name"]] = create_rtdl_mlp_skorch

from configs.model_configs.resnet_config import *
keyword = "resnet"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = create_resnet_regressor_skorch
model_keyword_dic[config_classif["model_name"]] = create_resnet_skorch

from configs.model_configs.ft_transformer_config import *
keyword = "ft_transformer"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = create_ft_transformer_regressor_skorch
model_keyword_dic[config_classif["model_name"]] = create_ft_transformer_skorch

from configs.model_configs.saint_config import *
keyword = "saint"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]] = SAINT
model_keyword_dic[config_classif["model_name"]] = SAINT


