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


def convert_keyword_to_function(keyword):
    if keyword == 'mlp_skorch_regressor':
        return create_mlp_skorch_regressor
    elif keyword == "uniform_data":
        return generate_uniform_data
    elif keyword == "periodic_triangle":
        return periodic_triangle