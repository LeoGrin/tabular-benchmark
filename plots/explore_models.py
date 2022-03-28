import streamlit as st
from utils.plot_utils import plot_decision_boudaries
import pickle
import numpy as np
import torch
from config import config
from utils.skorch_utils import create_mlp_skorch

st.title('Decision boundaries')

config_keyword = st.text_input("Config keyword")

model_names = st.text_input("Model name")

dataset = st.radio("Dataset", ('electricity', "wine", "california", "covtype"))


with open('saved_models/{}/{}/{}'.format(config_keyword, dataset, "data"), "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

#indices_train = np.random.choice(np.array(list(range(x_train.shape[0]))), size=2000, replace=True)
#indices_test = np.random.choice(np.array(list(range(x_test.shape[0]))), size=2000, replace=True)
x_train_sample = x_train#[indices_train]
x_test_sample = x_test#[indices_test]
y_train_sample = y_train#[indices_train]
y_test_sample = y_test#[indices_test]



x_min_init, x_max_init = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min_init, y_max_init = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

#x_min_init, x_max_init, y_min_init, y_max_init = x_min_init.astype(np.float32), x_max_init.astype(np.float32), y_min_init.astype(np.float32), y_max_init.astype(np.float32)

x_min, x_max = st.slider("x boudaries", float(x_min_init), float(x_max_init), value=(float(x_min_init), float(x_max_init)))
y_min, y_max = st.slider("y boudaries", float(y_min_init), float(y_max_init), value=(float(y_min_init), float(y_max_init)))



for model_name in model_names.split(","):
    with open("saved_models/{}/{}/{}_{}".format(config_keyword, dataset, model_names, "params"), "rb") as f:
        params = pickle.load(f)
    params.pop("method")
    params.pop("method_name")
    params["device"] = "cpu"
    model = create_mlp_skorch(id=None, wandb_run=None,
                              module__input_size=x_train.shape[1],
                              module__output_size=2,
                              **params)
    model.initialize()
    #model.module_.load_state_dict(torch.load("saved_models/{}/{}/{}.pt".format(config_keyword, dataset, model_name), map_location=torch.device('cpu')))
    model.load_params(f_params="saved_models/{}/{}/{}.pkl".format(config_keyword, dataset, model_name))
    st.pyplot(plot_decision_boudaries(x_train_sample, y_train_sample, x_test_sample, y_test_sample, model, model_name, x_min, x_max, y_min, y_max))



