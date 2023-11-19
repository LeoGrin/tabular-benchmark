import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pandas as pd
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from sklearn.preprocessing import LabelEncoder
import openml
import pickle

def balance_data(x, y):
    rng = np.random.RandomState(0)
    print("Balancing")
    print(x.shape)
    indices = [(y == i) for i in np.unique(y)]
    sorted_classes = np.argsort(
        list(map(sum, indices)))  # in case there are more than 2 classes, we take the two most numerous

    n_samples_min_class = sum(indices[sorted_classes[-2]])
    print("n_samples_min_class", n_samples_min_class)
    indices_max_class = rng.choice(np.where(indices[sorted_classes[-1]])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices[sorted_classes[-2]])[0]
    total_indices = np.concatenate((indices_max_class, indices_min_class))
    y = y[total_indices]
    indices_first_class = (y == sorted_classes[-1])
    indices_second_class = (y == sorted_classes[-2])
    y[indices_first_class] = 0
    y[indices_second_class] = 1

    return x.iloc[total_indices], y

def import_openml_data_no_transform(keyword, regression=False, categorical=False, rng=None):
    # keyword should be the openml task id
    task = openml.tasks.get_task(keyword)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    if not categorical:
        assert categorical_indicator is None or not np.array(categorical_indicator).astype(bool).any(), "There are categorical features in the dataset"
        categorical_indicator = None #easier to deal with
        if not regression:
            y = y.astype(np.int64)
    else:
        categorical_indicator = np.array(categorical_indicator).astype(bool)
        if not regression:
            y = y.astype(np.int64)
    return X, y, categorical_indicator

def import_openml_data_no_transform_dataset(keyword, regression=False, categorical=False, rng=None):
    # keyword should be the openml task id
    dataset = openml.datasets.get_dataset(keyword)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    if not categorical:
        assert categorical_indicator is None or not np.array(categorical_indicator).astype(bool).any(), "There are categorical features in the dataset"
        categorical_indicator = None #easier to deal with
        if not regression:
            y = y.astype(np.int64)
    else:
        categorical_indicator = np.array(categorical_indicator).astype(bool)
        if not regression:
            y = y.astype(np.int64)
    return X, y, categorical_indicator


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


def generate_uniform_data(n_samples,
                          n_features,
                          regression=None,
                          rng=None) -> np.array:
    """
    :param num_samples:
    :param num_features:
    :return:
    """
    return rng.uniform(low=-2, high=2, size=(n_samples, n_features))


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

