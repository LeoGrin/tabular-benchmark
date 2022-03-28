import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from target_function_classif import *
from data_transforms import *
from generate_data import *
from utils.streamlit import *
from utils.utils import remove_keys_from_dict, merge_dics
from generate_dataset_pipeline_streamlit import generate_dataset
from config import config, merge_configs
from utils.plot_utils import plot_comparison
import matplotlib
import bokeh

# st.set_page_config(layout='wide')

st.title('Data visualisation')




@st.cache
def load_results(path):
    return pd.read_csv(path)

config = st.sidebar.text_input("config keyword")

data_generation_functions, target_generation_functions, data_transforms_functions = merge_configs([config])
#if type(data_transforms_functions[0]) == list:
 #   data_transforms_functions = [dic for l in data_transforms_functions for dic in l]

do_generate = st.sidebar.radio("Generate data", ["Yes", "No"])

st.sidebar.title("Parameters")

reload_data = st.sidebar.button("Reload data")

st.sidebar.title("X generating process")
data_generation_name = st.sidebar.radio("Choose", np.unique([dic["method_name"] for dic in data_generation_functions]))
method = [dic["method"] for dic in data_generation_functions if dic["method_name"] == data_generation_name][0]
dics = [remove_keys_from_dict(dic, ["method", "method_name"]) for dic in data_generation_functions if dic["method_name"] == data_generation_name]
if len(dics) > 1:
    dic = merge_dics(dics)
else:
    dic = dics[0]
data_generation_dic = create_widgets_from_param_dic(dic)
data_generation_dic["method"] = method
data_generation_dic["method_name"] = data_generation_name

if data_generation_name != "open_ml":
    st.sidebar.title("Y generating process")
    target_generation_name = st.sidebar.radio("Choose", np.unique([dic["method_name"] for dic in target_generation_functions]))
    method = [dic["method"] for dic in target_generation_functions if dic["method_name"] == target_generation_name][0]
    dics = [remove_keys_from_dict(dic, ["method", "method_name"]) for dic in target_generation_functions if dic["method_name"] == target_generation_name]
    if len(dics) > 1:
        dic = merge_dics(dics)
    else:
        dic = dics[0]
    target_generation_dic = create_widgets_from_param_dic(dic)
    target_generation_dic["method"] = method
    target_generation_dic["method_name"] = target_generation_name
else:
    target_generation_dic = {"method":None, "method_name":"no_transform"} #TODO should be no_target

st.sidebar.title("Data transforms")
data_transforms_dic_list = []
n_transforms = max([len(l) for l in data_transforms_functions])
transform_name_list = []
for i in range(n_transforms):
    st.sidebar.title("Tranform {}".format(i))
    transforms_list = [l[i] for l in data_transforms_functions if len(l) > i]
    transform_name_list.append(st.sidebar.radio("Choose transform {}".format(i), np.unique([dic["method_name"] for dic in transforms_list])))
    if transform_name_list[i] != "no_transform":
        method = [dic["method"] for dic in transforms_list if dic["method_name"] == transform_name_list[i]][0]
        dics = [remove_keys_from_dict(dic, ["method", "method_name"]) for dic in transforms_list if dic["method_name"] == transform_name_list[i]]
        if len(dics) > 1:
            dic = merge_dics(dics)
        else:
            dic = dics[0]
        data_transforms_dic = create_widgets_from_param_dic(dic)
        data_transforms_dic["method"] = method
        data_transforms_dic["method_name"] = transform_name_list[i]
        data_transforms_dic_list.append(data_transforms_dic)
    else:
        if i == 0:
            data_transforms_dic_list.append({"method": None, "method_name": "no_transform"}) #TODO this is weird, but needed right now because we don't deal with empty list in config
        else:
            data_transforms_dic_list.append({"method":None, "method_name":np.nan})


# data_transforms_dic_list = []
# methods = [dic["method"] for dic in data_transforms_functions if st.sidebar.checkbox(dic["method_name"])]
# dics = [remove_key_from_dict(dic, "method") for dic in target_generation_functions if dic["method_name"] == target_generation_name]
# if len(dics) > 1:
#     dic = merge_dics(dics)
# else:
#     dic = dics[0]
# for dic in data_transforms_functions:
#     if :
#         dic_to_add = create_widgets_from_param_dic(remove_key_from_dict(dic, "method"))
#         dic_to_add["method"] = dic["method"]
#         dic_to_add["method_name"] = dic["method_name"]
#         data_transforms_dic_list.append(dic_to_add)


if "seed" not in st.session_state.keys():
    seed = np.random.randint(0, 1e8)
    st.session_state["seed"] = seed
else:
    if reload_data:
        seed = np.random.randint(0, 1e8)
        st.session_state["seed"] = seed
    else:
        seed = st.session_state["seed"]

@st.cache
def random_subset(x, y, max_num_samples):
    if max_num_samples != "all":
        indices = np.random.choice(range(x.shape[0]), min(max_num_samples, x.shape[0]), replace=False)
        x = x[indices, :]
        y = y[indices]
    return x, y

#@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
#def mk_figure(fig, dim_1, dim_2, x, y, colors):
#    fig.scatter(x[:, dim_1], x[:, dim_2], color=colors[y], alpha=0.6, s=4)

rng = np.random.RandomState(seed)

if do_generate == "Yes":
    data = generate_dataset([data_generation_dic, target_generation_dic, data_transforms_dic_list], rng, seed)
    if data is None:
        st.error("Data could not be generated")
    else:
        x, y = data
        max_num_samples = st.select_slider("Display only a subset of samples", [1000, 10000, x.shape[0]], value = 10000)
        x, y = random_subset(x, y, max_num_samples)
        y = y.astype(int)


        st.text("Number of samples: {}".format(x.shape[0]))
        st.text("Number of features: {}".format(x.shape[1]))
        columns = st.columns(2)
        n_rows=2
        figs = []
        for i in range(len(columns)):
            l = []
            for j in range(n_rows):
                l.append(plt.subplots())
            figs.append(l)
        colors = np.array(["red", "blue"])

        for i, col in enumerate(columns):
            for j in range(n_rows):
                dim_1 = col.slider("x dim, {},{}".format(i, j), min_value=0, max_value=x.shape[1] - 1)
                dim_2 = col.slider("y dim, {},{}".format(i, j), min_value=0, max_value=x.shape[1] - 1)
                figs[i][j][1].scatter(x[:, dim_1], x[:, dim_2], color=colors[y], alpha=0.6, s=4)
                col.pyplot(figs[i][j][0])



st.title("Model comparison")
try:
    df = load_results("results/{}.csv".format(config))
    print(df)

    fig = plot_comparison(df, remove_keys_from_dict(data_generation_dic, ["method"]),
                              remove_keys_from_dict(target_generation_dic, ["method"]),
                              [remove_keys_from_dict(data_transforms_dic, ["method"]) for data_transforms_dic in data_transforms_dic_list],
                              ["mlp_skorch", "gbt", "rf"])
    st.pyplot(fig)
except Exception as e:
    print(e)
    #s = str(e)
    #st.error(s)
    st.error("Could not load results")





