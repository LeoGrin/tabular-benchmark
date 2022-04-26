from utils.utils import numpy_to_dataset, remove_keys_from_dict
import streamlit as st
import numpy as np

#There are three steps to generate a dataset:
# 1) Generate x
# 2) Generate y
# 3) Transform x
# For each of this step, the user can provide the name of the method used (or a list of method for step 3),
# and a dictionary (or a list of dictionaries for step 3) with the different parameters


@st.cache(allow_output_mutation=True)
def generate_data(method, parameters, rng, seed=1):
    x = method(**parameters, rng=rng)
    return x

@st.cache
def generate_target(x, method, parameters, rng, seed=1):
    y = method(x, **parameters, rng=rng)
    return y

@st.cache
def transform_data(x, y, methods, parameters, rng, seed=1):
    assert len(methods) == len(parameters)
    for i in range(len(methods)):
        if not methods[i] is None:
            x, y = methods[i](x, y, **parameters[i], rng=rng)
    return x, y

def generate_dataset(settings, rng, seed=1):
    #TODO no duplicate with streamlit function
    #right now the seed is only for streamlit
    generate_method, target_method = [settings[i]["method"] for i in range(2)]
    transform_methods = [transform_params["method"] for transform_params in settings[2]]
    transform_parameters_list = [remove_keys_from_dict(transform_params, ["method", "method_name"]) for transform_params in settings[2]]
    generate_parameters, target_parameters = [remove_keys_from_dict(settings[i], ["method", "method_name"]) for i in range(2)]
    data = generate_data(generate_method, generate_parameters, rng, seed)
    if data is None:
        return None
    if len(data) == 2: #if generate data returns x, y #TODO something cleaner
        x, y = data
        x = x.astype(np.float64)
    else:
        x = data
        x = x.astype(np.float64)
        y = generate_target(x, target_method, target_parameters, rng, seed)
    x, y = transform_data(x, y, transform_methods, transform_parameters_list, rng, seed)
    return x, y
