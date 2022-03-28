import numpy as np
from scipy.stats import special_ortho_group
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def marginal_transformations(x, y, function, vectorized=False, rng=None):
    """
    Transform each feature of x, either by different functions or by the same
    function.
    :param x: numpy array to transform, of shape (num_samples, num_features)
    :param function: functions to apply to the marginals of x
    If it's a function, apply this function to all columns of x
    If it's a list of function of len num_features, apply function i to feature i
    :param vectorized x: can the function be applied to x, or should it be applied
    column per column
    :return: a new array with each marginal transformed
    """
    if type(function) == list or type(function) == np.array:
        new_x = np.ones_like(x)
        for i in range(x.shape[1]):
            new_x[:, i] = function[i](x[:, i])
    else:
        if vectorized:
            new_x = function(x)
        else:
            new_x = np.ones_like(x)
            for i in range(x.shape[1]):
                new_x[:, i] = function(x[:, i])

    return new_x, y


def apply_random_rotation(x_train, x_test, y_train, y_test, rng=None):
    num_samples, num_features = x_train.shape
    rotation_matrix = special_ortho_group.rvs(num_features, random_state=rng)
    return x_train @ rotation_matrix, x_test @ rotation_matrix, y_train, y_test

def add_noise(x, y, noise_type="white", scale=1, rng=None):
    if noise_type != "white":
        raise ValueError("Only white noise is supported right now")
    x += rng.multivariate_normal(mean=np.zeros(x.shape[1]), cov=scale * np.diag(
        ((np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25)) / 1.349) ** 2), size=x.shape[0])
    return x, y


def add_uninformative_features(x, y, num_uninformatives=1, rng=None):
    """
    Add num_uninformatives uninformative gaussain columns to x, imitating the mean and interquartile range of
    a randomly chosen subset of x columns
    :param x: data to which the features are added
    :param num_uninformatives: number of uninformative columns to add
    :return: x with uninformative features added
    """
    # TODO add correlations
    num_samples, num_features = x.shape
    cols_to_imitate = rng.choice(range(num_features),
                                 num_uninformatives)  # columns whose mean and interquartile we'll imitate in the uninformative columns
    new_features = rng.multivariate_normal(mean=np.mean(x, axis=0)[cols_to_imitate],
                                           cov=np.diag(((np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25))[
                                                            cols_to_imitate] / 1.349) ** 2),
                                           size=num_samples)  # for a gaussian, interquartile range is 1.349σ
    return np.concatenate((x, new_features), axis=1), y


def gaussienize(x_train, x_test, y_train, y_test, type="standard", rng=None):
    """
    Gaussienize the data
    :param x: data to transform
    :param type: {"standard","robust", "quantile", "power", "quantile_uniform"}
    :return: the transformed data
    """
    if type == "standard":
        t = StandardScaler()
    elif type == "robust":
        t = RobustScaler()
    elif type == "quantile":
        t = QuantileTransformer(output_distribution="normal", random_state=rng)
    elif type == "quantile_uniform":
        t = QuantileTransformer(output_distribution="uniform", random_state=rng)
    elif type == "power":
        t = PowerTransformer(random_state=rng)

    x_train = t.fit_transform(x_train)
    x_test = t.transform(x_test)

    return x_train, x_test, y_train, y_test


def cluster_1d(x, y, type="kmeans", rng=None, **kwargs):
    if type == "kmeans":
        for i in range(x.shape[1]):
            x[:, i] = KMeans(random_state=rng, **kwargs).fit_predict(x[:, i].reshape(-1, 1))
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)

    else:
        raise ValueError("Not implemented yet")

    return x, y


def select_features_rf(x_train, x_test, y_train, y_test, rng, num_features=None, importance_cutoff=None,
                       return_features=False):
    assert (num_features is None) + (importance_cutoff is None) == 1  # xor
    if isinstance(num_features, float):
        num_features = int(num_features * x_train.shape[1])
    rf = RandomForestClassifier(random_state=rng)
    rf.fit(x_train, y_train)
    if importance_cutoff is not None:
        num_features = max(1, np.sum(rf.feature_importances_ > importance_cutoff)) # At least one feature

    x_train = x_train[:, np.argsort(rf.feature_importances_)[-num_features:]]
    x_test = x_test[:, np.argsort(rf.feature_importances_)[-num_features:]]
    if not return_features:
        return x_train, x_test, y_train, y_test
    else:
        return x_train, x_test, y_train, y_test, np.argsort(rf.feature_importances_)[-num_features:]

def remove_features_rf(x_train, x_test, y_train, y_test, rng, num_features_to_remove=None, importance_cutoff=None,
                       return_features=False):
    assert (num_features_to_remove is None) + (importance_cutoff is None) == 1  # xor
    assert x_train.shape[1] == x_test.shape[1]
    if isinstance(num_features_to_remove, float):
        num_features_to_remove = int(num_features_to_remove * x_train.shape[1])
    rf = RandomForestClassifier(random_state=rng)
    rf.fit(x_train, y_train)
    if importance_cutoff is not None:
        num_features_to_remove = min(x_train.shape[1] - 1, np.sum(rf.feature_importances_ < importance_cutoff)) # At least one feature
    features_to_keep = np.argsort(rf.feature_importances_)[- (x_train.shape[1] - num_features_to_remove):]
    x_train = x_train[:, features_to_keep]
    x_test = x_test[:, features_to_keep]
    assert x_train.shape[1] == x_test.shape[1]
    if not return_features:
        return x_train, x_test, y_train, y_test
    else:
        return x_train, x_test, y_train, y_test, np.argsort(rf.feature_importances_)[-(x_train.shape[1] - num_features_to_remove):]


def tree_quantile_transformer(x_train, x_test, y_train, y_test, regression=False, normalize=True, rng=None):
    qt = QuantileTransformer(output_distribution="uniform", random_state=rng)
    x_train_ = qt.fit_transform(x_train)  # between 0 and 1
    x_test_ = qt.transform(x_test)
    if regression:
        tree = DecisionTreeRegressor(random_state=rng)
    else:
        tree = DecisionTreeClassifier(random_state=rng)
    for i in range(x_train_.shape[1]):
        tree.fit(x_train_[:, i].reshape(-1, 1), y_train)
        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds != -2]
        thresholds = np.sort(thresholds)[::-1]
        # thresholds = np.array([0] + list(thresholds) + [1])
        thresholds_belonging_train = np.floor(x_train_[:, i] * len(thresholds)).astype(
            np.int) - 1  # to which threshold does each value belong
        x_train_[:, i] = thresholds[thresholds_belonging_train] + (x_train_[:, i] % len(thresholds)) * (
                    thresholds[1] - thresholds[0])
        thresholds_belonging_test = np.floor(x_test_[:, i] * len(thresholds)).astype(
            np.int) - 1  # to which threshold does each value belong
        x_test_[:, i] = thresholds[thresholds_belonging_test] + (x_test_[:, i] % len(thresholds)) * (
                    thresholds[1] - thresholds[0])
    if normalize:
        x_train_ -= np.mean(x_train_, axis=0)
        x_train_ /= np.std(x_train_, axis=0)
        x_test_ -= np.mean(x_test_, axis=0)
        x_test_ /= np.std(x_test_, axis=0)
    return x_train_, x_test_, y_train, y_test


def remove_pseudo_categorial(x, y, threshold, rng):
    """
    Remove features with too few values
    :param x: data to be transformed
    :param threshold: if int, the number of unique values under which we remove the feature. If float, the proportion.
    :return:
    """
    features_to_remove = []
    for i in range(x.shape[1]):
        n = len(np.unique(x[:, i]))
        if isinstance(threshold, (int, np.integer)) or (isinstance(threshold, (float, np.float)) and threshold > 1):
            if n < threshold:
                features_to_remove.append(i)
        elif isinstance(threshold, (float, np.float)):
            if (n / x.shape[0]) < threshold:
                features_to_remove.append(i)
        else:
            raise ValueError("threshold should be int or float")
    features_to_remove = np.array(features_to_remove)
    if len(features_to_remove):
        return np.delete(x, features_to_remove, axis=1), y
    else:
        return x, y


def remove_last_column(x, y, rng):
    return x[:, :-1], y


def balance(x_train, x_test, y_train, y_test, rng):
    indices_train = [(y_train == 0), (y_train == 1)]
    max_class = np.argmax(list(map(sum, indices_train)))
    min_class = np.argmin(list(map(sum, indices_train)))
    n_samples_min_class = sum(indices_train[min_class])
    indices_max_class = rng.choice(np.where(indices_train[max_class])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices_train[min_class])[0]
    total_indices_train = np.concatenate((indices_max_class, indices_min_class))

    indices_test = [(y_train == 0), (y_train == 1)]
    max_class = np.argmax(list(map(sum, indices_test)))
    min_class = np.argmin(list(map(sum, indices_test)))
    n_samples_min_class = sum(indices_test[min_class])
    indices_max_class = rng.choice(np.where(indices_test[max_class])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices_test[min_class])[0]
    total_indices_test = np.concatenate((indices_max_class, indices_min_class))
    return x_train[total_indices_train], x_test[total_indices_test], y_train[total_indices_train], y_test[
        total_indices_test]


def limit_size(x, y, n_samples, rng):
    indices = list(range(len(y)))
    chosen_indices = rng.choice(indices, n_samples, replace=False)
    return x[chosen_indices], y[chosen_indices]


if __name__ == """__main__""":
    print(2)
