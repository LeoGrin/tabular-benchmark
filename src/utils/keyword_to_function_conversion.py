from target_function_classif import *
from generate_data import *
from data_transforms import *


convert_keyword_to_function = {
    "uniform_data": generate_uniform_data,
    "periodic_triangle": periodic_triangle,
    "openml_no_transform": import_openml_data_no_transform,
    "openml_no_transform_dataset": import_openml_data_no_transform_dataset,
    "gaussienize": gaussienize,
    "select_features_rf": select_features_rf,
    "remove_features_rf": remove_features_rf,
    "remove_useless_features": remove_useless_features,
    "add_uninformative_features": add_uninformative_features,
    "random_rotation": apply_random_rotation,
    "remove_high_frequency_from_train": remove_high_frequency_from_train,
    "normalize_y": normalize_y,
    "no_transform": None
}
