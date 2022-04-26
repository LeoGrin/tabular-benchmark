import streamlit as st
import numpy as np
import numbers

def create_widget_from_params_values(param_name, values):
    #print("valuesuuuu",values)
    if type(values) == list or type(values) == np.ndarray:
        if isinstance(values[0], numbers.Number):
            if len(values) == 1:
                st.text("{}: {}".format(param_name, values[0]))
                return values[0]
            return st.sidebar.select_slider(param_name, np.sort(values))
        if type(values[0]) == str or type(values[0]) == np.str_:
            return st.sidebar.radio(param_name, values)
    elif isinstance(values, (str, np.str)):
        return st.sidebar.text_input(param_name, value=values)
    elif isinstance(values, (int, np.integer)):
        return st.sidebar.number_input(param_name, step=1, value=values)
    elif isinstance(values, (float, np.float)):
        return st.sidebar.number_input(param_name, value=values)

def create_widgets_from_param_dic(dic):
    #print("#########")
    #print(dic)
    widget_dic = {}
    for key in dic.keys():
        widget_dic[key] = create_widget_from_params_values(key, dic[key])
    #print(widget_dic)
    #print("@@@@@@@@@@@")
    return widget_dic
