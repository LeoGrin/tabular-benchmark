from utils.utils import numpy_to_dataset, remove_keys_from_dict
import numpy as np
from utils.keyword_to_function_conversion import convert_keyword_to_function
from sklearn.model_selection import train_test_split

#There are three steps to generate a dataset:
# 1) Generate x
# 2) Generate y
# 3) Transform x
# For each of this step, the user can provide the name of the method used (or a list of method for step 3),
# and a dictionary (or a list of dictionaries for step 3) with the different parameters


def generate_data(config, rng):
    method = convert_keyword_to_function(config["data__method_name"])
    data_config = {}
    for key in config.keys():
        if key.startswith("data__") and key != "data__method_name":
            data_config[key[len("data__"):]] = config[key]
    data = method(**data_config, rng=rng)
    return data

def generate_target(x, config, rng):
    method = convert_keyword_to_function(config["target__method_name"])
    target_config = {}
    for key in config.keys():
        if key.startswith("target__") and key != "target__method_name":
            target_config[key[len("target__"):]] = config[key]
    data = method(x, **target_config, rng=rng)
    return data

def transform_data(x_train, x_test, y_train, y_test, config, rng):
    i = 0
    while True:
        if f"transform__{i}__method_name" in config.keys():
            method = convert_keyword_to_function(config[f"transform__{i}__method_name"])
            target_config = {}
            for key in config.keys():
                if key.startswith(f"transform__{i}__") and key != "transform__{i}__method_name":
                    target_config[key[len(f"transform__{i}__"):]] = config[key]
                    x_train, x_test, y_train, y_test = method(x_train, x_test, y_train, y_test, **target_config, rng=rng)
        else:
            break
        i += 1

    return x_train, x_test, y_train, y_test

def data_to_train_test(x, y, config, rng=None):
    #TODO: add the possibility for Cross Validation
    n_rows = x.shape[0]
    if not config["max_train_samples"] is None:
        train_set_prop = min(config["max_train_samples"] / n_rows, config["train_set_prop"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= train_set_prop, random_state=rng)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rng)
    return x_train, x_test, y_train, y_test


def generate_dataset(config, rng):
    data = generate_data(config, rng)
    if data is None:
        return None
    if len(data) == 2: #if generate data returns x, y #TODO something cleaner
        x, y = data
        x = x.astype(np.float64)
    else:
        x = data
        x = x.astype(np.float64)
        y = generate_target(x, config, rng)

    x_train, x_test, y_train, y_test = data_to_train_test(x, y, config, rng=rng)

    x_train, x_test, y_train, y_test = transform_data(x_train, x_test, y_train, y_test, config, rng)
    return x_train, x_test, y_train, y_test
