import openml
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
openml.config.cache_directory = os.path.expanduser(os.getcwd() + "/openml_cache")

def get_metadata(suite_id, output_name):
    df = pd.DataFrame()
    dataset_names = []
    n_samples_list = []
    n_features_list = []
    new_link_list = []
    benchmark_suite = openml.study.get_suite(suite_id)  # obtain the benchmark suite
    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        print(f"dataset {dataset.name}")
        # retrieve categorical data for encoding
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        dataset_names.append(dataset.name)
        n_samples_list.append(X.shape[0])
        n_features_list.append(X.shape[1])
        new_link_list.append("https://www.openml.org/d/{}".format(dataset.id))
    df["dataset_name"] = dataset_names
    df["n_samples"] = n_samples_list
    df["n_features"] = n_features_list
    df["new_link"] = new_link_list
    df.to_csv(output_name, index=False)

def save_suite(suite_id, dir_name, save_categorical_indicator=False, regression=True):
    benchmark_suite = openml.study.get_suite(suite_id)  # obtain the benchmark suite
    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        print(f"Downloading dataset {dataset.name}")
        # retrieve categorical data for encoding
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = np.array(X).astype(np.float32)
        if regression:
            y = np.array(y).astype(np.float32)
        else:
            le = LabelEncoder()
            y = le.fit_transform(np.array(y))
        with open("{}/data_{}".format(dir_name, dataset.name), "wb") as f:
            if save_categorical_indicator:
                pickle.dump((X, y, categorical_indicator), f)
            else:
                pickle.dump((X, y), f)
                
suites_id = {"numerical_regression": 297,
          "numerical_classification": 298,
          "categorical_regression": 299,
          "categorical_classification": 304}

print("Saving datasets from suite: {}".format("numerical_regression"))
get_metadata(suites_id["numerical_regression"], "numerical_regression.csv")
#save_suite(suites_id["numerical_regression"],
#           "data/numerical_only/regression",
#           save_categorical_indicator=False)

print("Saving datasets from suite: {}".format("numerical_classification"))
get_metadata(suites_id["numerical_classification"], "numerical_classification.csv")

#save_suite(suites_id["numerical_classification"],
#           "data/numerical_only/balanced",
#           save_categorical_indicator=False,
#           regression=False)

print("Saving datasets from suite: {}".format("categorical_regression"))
get_metadata(suites_id["categorical_regression"], "categorical_regression.csv")

#save_suite(suites_id["categorical_regression"],
#           "data/num_and_cat/regression",
#           save_categorical_indicator=True)

print("Saving datasets from suite: {}".format("categorical_classification"))
get_metadata(suites_id["categorical_classification"], "categorical_classification.csv")

#save_suite(suites_id["categorical_classification"],
#           "data/num_and_cat/balanced",
#           save_categorical_indicator=True,
#           regression=False)