import openml
import numpy as np
import pickle

#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key

def save_suite(suite_id, dir_name, save_categorical_indicator=False):
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
        y = np.array(y).astype(np.int32)
        with open("{}/data_{}".format(dir_name, dataset.name), "wb") as f:
            if save_categorical_indicator:
                pickle.dump((X, y, categorical_indicator), f)
            else:
                pickle.dump((X, y), f)
                
suites_id = {"numerical_regression": 297,
          "numerical_classification": 298,
          "categorical_regression": 299,
          "categorical_classification": 300}

print("Saving datasets from suite: {}".format("numerical_regression"))
save_suite(suites_id["numerical_regression"],
           "data/numerical_only/regression",
           save_categorical_indicator=False)

print("Saving datasets from suite: {}".format("numerical_classification"))
save_suite(suites_id["numerical_classification"],
           "data/numerical_only/balanced",
           save_categorical_indicator=False)

print("Saving datasets from suite: {}".format("categorical_regression"))
save_suite(suites_id["categorical_regression"],
           "data/num_and_cat/regression",
           save_categorical_indicator=True)

print("Saving datasets from suite: {}".format("categorical_classification"))
save_suite(suites_id["categorical_classification"],
           "data/num_and_cat/balanced",
           save_categorical_indicator=True)