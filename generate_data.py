import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pandas as pd
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from sklearn.preprocessing import LabelEncoder
import openml
import pickle


def import_open_ml_data(openml_task_id=None, path_to_dir="openML_data", max_num_samples=None, rng=None) -> pd.DataFrame:
    """
    :param int openml_task_id:
    :param path_to_file:
    :return:
    """
    if openml_task_id is None:
        raise ValueError('Not implemented yet')

    if not path_to_dir is None:
        with open("{}/openML_data_task_{}".format(path_to_dir, openml_task_id), "rb") as f:
            X, y = pickle.load(f)
    elif path_to_dir is None:
        print("No saved file, downloading the data for this task")
        task = openml.tasks.get_task(openml_task_id)  # download the OpenML task
        dataset = task.get_dataset()
        # retrieve categorical data for encoding
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        categorical_indicator = np.array(categorical_indicator)
        print("{} categorical columns".format(sum(categorical_indicator)))
        print("{} columns".format(X.shape[1]))
        y_encoder = LabelEncoder()
        # remove missing values
        missing_rows_mask = X.isnull().any(axis=1)
        if sum(missing_rows_mask) > X.shape[0] / 5:
            print("Removed {} rows with missing values on {} rows".format(
                sum(missing_rows_mask), X.shape[0]))
        X = X[~missing_rows_mask]
        y = y[~missing_rows_mask]

        n_rows_non_missing = X.shape[0]
        if n_rows_non_missing == 0:
            print("Removed all rows")
            return None

        print("removing {} categorical features among {} features".format(sum(categorical_indicator), X.shape[1]))
        X = X.to_numpy()[:, ~categorical_indicator]  # remove all categorical columns
        if X.shape[1] == 0:
            print("removed all features, skipping this task")
            return None

        y = y_encoder.fit_transform(y)

    if not (max_num_samples is None):
        # max_num_samples = int(max_num_samples)
        if max_num_samples < X.shape[0]:
            indices = rng.choice(range(X.shape[0]), max_num_samples, replace=False)
            X = X[indices]
            y = y[indices]
    return X, y


def import_real_data(keyword=None, balanced=True, path_to_dir="data", max_num_samples=None, regression=False, dim=[],
                     rng=None):
    if regression:
        with open("{}/numerical_only/regression/full/data_{}".format(path_to_dir, keyword), "rb") as f:
            X, y = pickle.load(f)
            if len(dim) > 0:
                print("selecting dims")
                with open("{}/numerical_only/names_{}".format(path_to_dir, keyword), "rb") as g:
                    names = pickle.load(g)
                    X = X[:, [names.index(i) for i in dim]]
    else:
        if balanced:
            with open("{}/numerical_only/balanced/data_{}".format(path_to_dir, keyword), "rb") as f:
                X, y = pickle.load(f)
        else:
            with open("{}/numerical_only/full/data_{}".format(path_to_dir, keyword), "rb") as f:
                X, y = pickle.load(f)
    # handled in run_experiment2 now
    # if not (max_num_samples is None):
    #     if max_num_samples < X.shape[0]:
    #         indices = rng.choice(range(X.shape[0]), max_num_samples, replace=False)
    #         X = X[indices]
    #         y = y[indices]
    return X, y


def generate_synthetic_data(num_samples,
                            num_dimensions,
                            generation_function,
                            rng,
                            **kwargs) -> np.array:
    """
    :param num_samples:
    :param num_dimensions:
    :param base_data_generation:
    :return:
    """
    # TODO RNG
    data_shape = (num_samples, num_dimensions)
    return generation_function(size=data_shape, **kwargs)


def generate_gaussian_data(num_samples,
                           num_features,
                           cov_matrix="identity",
                           regression=None,
                           rng=None) -> np.array:
    """
    :param num_samples:
    :param num_features:
    :param cov_matrix: One of these
    "diagonal"
    "random"
    "random_sparse"
    "random_sparse_precision"
    :return:
    """
    if cov_matrix == "identity":  # TODO add a parameter to control variance
        return rng.multivariate_normal(mean=np.zeros(num_features), cov=np.eye(num_features),
                                       size=num_samples)
    elif cov_matrix == "random":
        return rng.multivariate_normal(mean=np.zeros(num_features), cov=make_spd_matrix(num_features),
                                       size=num_samples)
    elif cov_matrix == "random_sparse":
        return rng.multivariate_normal(mean=np.zeros(num_features), cov=make_sparse_spd_matrix(num_features),
                                       size=num_samples)
    elif cov_matrix == "random_sparse_precision":
        precision_matrix = make_sparse_spd_matrix(num_features)
        cov_matrix = np.linalg.inv(precision_matrix)
        return rng.multivariate_normal(mean=np.zeros(num_features), cov=cov_matrix,
                                       size=num_samples)


def generate_uniform_data(num_samples,
                          num_features,
                          regression=None,
                          rng=None) -> np.array:
    """
    :param num_samples:
    :param num_features:
    :return:
    """
    return rng.uniform(low=-2, high=2, size=(num_samples, num_features))


def generate_student_data(num_samples,
                          num_features,
                          df=1,
                          rng=None) -> np.array:
    """
    :param num_samples:
    :param num_features:
    :return:
    """
    # TODO other choices that independent
    return rng.standard_t(df=df, size=(num_samples, num_features))


def generate_periodic_triangles_uniform(num_samples, period=None, offset=None, period_size=None, noise=True, regression=None, rng=None):
    """
    Generate periodic triangles, with ~the same number of point per period (uniform in each period)
    :param num_samples:
    :return:
    """
    # Conversions
    if not period_size is None:
        if not period is None:
            offset = (4 - period_size * period) / 2
        elif not offset is None:
            period = (4 - offset * 2) / period_size  # FIXME
    assert offset < 2, "Too many period, or too big offset"

    x = np.zeros((num_samples))
    num_samples_in_offset_zone = int(
        num_samples * (offset * 2 / 4))  # to match with the function used with a uniform(-2, 2) distrib
    x[:num_samples_in_offset_zone // 2] = rng.uniform(low=-2, high=-2 + offset, size=num_samples_in_offset_zone // 2)
    x[num_samples_in_offset_zone // 2:num_samples_in_offset_zone] = rng.uniform(low=2 - offset, high=2,
                                                                                size=num_samples_in_offset_zone - num_samples_in_offset_zone // 2)

    num_samples_per_period = int((num_samples - num_samples_in_offset_zone) / period)
    for i in range(period):
        x[num_samples_in_offset_zone + i * num_samples_per_period:num_samples_in_offset_zone + (
                    i + 1) * num_samples_per_period] = rng.uniform(low=-2 + offset + i * period_size,
                                                                   high=-2 + offset + (i + 1) * period_size,
                                                                   size=num_samples_per_period)

    res = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] < -2 + offset or x[i] > 2 - offset:
            res[i] = 0
        else:
            res[i] = 2 * np.abs(np.abs(x[i]) / period_size - int(np.abs(x[i]) / period_size + 1 / 2))
            # res[i] = np.sin(x[i] * (2 * np.pi) / (4 - 2 * offset) * period)
    # res = np.sin(x.reshape(-1) * np.pi / period).astype(np.float32)
    if noise:
        res += rng.normal(0, 0.1, x.shape[0])
    return x.reshape(-1, 1), res


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    x, y = generate_periodic_triangles_uniform(1000, period=8, offset=None, period_size=0.3, noise=False, rng=rng)
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.show()

