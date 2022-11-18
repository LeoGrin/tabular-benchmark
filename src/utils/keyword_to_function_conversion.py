from target_function_classif import *
from generate_data import *
from data_transforms import *


convert_keyword_to_function = {
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
def create_stacking_classifier(base_estimator_keyword_list, final_estimator_keyword):
    base_estimators = []
    for base_estimator_keyword in base_estimator_keyword_list:
        base_estimators.append((base_estimator_keyword, convert_keyword_to_function[base_estimator_keyword]()))
    final_estimator = convert_keyword_to_function[final_estimator_keyword]()
    return StackingClassifier(base_estimators, final_estimator)

convert_keyword_to_function["stacking"] = create_stacking_classifier