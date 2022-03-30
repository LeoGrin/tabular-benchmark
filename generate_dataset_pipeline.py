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


def generate_data(method, parameters, rng):
    x = method(**parameters, rng=rng)
    return x

def generate_target(x, method, parameters, rng):
    y = method(x, **parameters, rng=rng)
    return y

def transform_data(x, y, methods, parameters, rng):
    assert len(methods) == len(parameters)
    for i in range(len(methods)):
        if not methods[i] is None:
            x, y = methods[i](x, y, **parameters[i], rng=rng)
    return x, y

def data_to_train_test(x, y, train_set_prop=0.75, max_train_samples=None, rng=None):
    #TODO: add the possibility for Cross Validation
    n_rows = x.shape[0]
    if not max_train_samples is None:
        train_set_prop = min(max_train_samples / n_rows, train_set_prop)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_set_prop, random_state=rng)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rng)
    return x_train, x_test, y_train, y_test

def generate_dataset(settings, rng):
    #TODO no duplicate with streamlit function
    #right now the seed is only for streamlit
    generate_method, target_method = [convert_keyword_to_function(settings[i]["method_name"]) for i in range(2)]
    #transform_methods = [transform_params["method"] for transform_params in settings[2]]
    #transform_parameters_list = [remove_keys_from_dict(transform_params, ["method", "method_name"]) for transform_params in settings[2]]
    generate_parameters, target_parameters = [remove_keys_from_dict(settings[i], ["method", "method_name"]) for i in range(2)] #FIXME
    data = generate_data(generate_method, generate_parameters, rng)
    if data is None:
        return None
    if len(data) == 2: #if generate data returns x, y #TODO something cleaner
        x, y = data
        x = x.astype(np.float64)
    else:
        x = data
        x = x.astype(np.float64)
        y = generate_target(x, target_method, target_parameters, rng)

    x_train, x_test, y_train, y_test = data_to_train_test(x, y, **settings[3], rng=rng)

    #x, y = transform_data(x, y, transform_methods, transform_parameters_list, rng)
    return x, y

def apply_transform(x_train, x_test, y_train, y_test, params_transform, rng):
    transform_methods = [convert_keyword_to_function(transform_params["method_name"]) for transform_params in params_transform]
    transform_parameters_list = [remove_keys_from_dict(transform_params, ["method", "method_name"]) for transform_params in params_transform]
    print("transform_methods")
    print(transform_methods)

    for i in range(len(transform_methods)):
            if not transform_methods[i] is None:
                print("transform")
                print(transform_methods[i])
                x_train, x_test, y_train, y_test = transform_methods[i](x_train, x_test, y_train, y_test, **transform_parameters_list[i], rng=rng)
    return x_train, x_test, y_train, y_test