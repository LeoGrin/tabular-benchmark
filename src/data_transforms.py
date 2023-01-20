import numpy as np
from scipy.stats import special_ortho_group
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from configs.all_model_configs import model_keyword_dic


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


def apply_random_rotation(x_train, x_val, x_test, y_train, y_val, y_test, deactivated=False, rng=None):
    if deactivated:
        return x_train, x_val, x_test, y_train, y_val, y_test
    num_samples, num_features = x_train.shape
    rotation_matrix = special_ortho_group.rvs(num_features, random_state=rng)
    return x_train @ rotation_matrix, x_val @ rotation_matrix, x_test @ rotation_matrix, y_train, y_val, y_test


def add_noise(x, y, noise_type="white", scale=1, rng=None):
    if noise_type != "white":
        raise ValueError("Only white noise is supported right now")
    x += rng.multivariate_normal(mean=np.zeros(x.shape[1]), cov=scale * np.diag(
        ((np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25)) / 1.349) ** 2), size=x.shape[0])
    return x, y


def add_uninformative_features(x_train, x_val, x_test, y_train, y_val, y_test, multiplier=2, rng=None):
    """
    Add num_uninformatives uninformative gaussian columns to x, imitating the mean and interquartile range of
    a randomly chosen subset of x columns
    :param x: data to which the features are added
    :param num_uninformatives: number of uninformative columns to add
    :return: x with uninformative features added
    """
    # TODO add correlations
    print("Adding uninformative features")
    num_uninformatives = int((multiplier - 1) * x_train.shape[1])
    if num_uninformatives == 0:
        return x_train, x_val, x_test, y_train, y_val, y_test
    num_samples_train, num_features = x_train.shape
    cols_to_imitate = rng.choice(range(num_features),
                                 num_uninformatives)  # columns whose mean and interquartile we'll imitate in the uninformative columns
    new_features_train = rng.multivariate_normal(mean=np.mean(x_train, axis=0)[cols_to_imitate],
                                                 cov=np.diag(((np.quantile(x_train, 0.75, axis=0) - np.quantile(x_train,
                                                                                                                0.25,
                                                                                                                axis=0))[
                                                                  cols_to_imitate] / 1.349) ** 2),
                                                 size=num_samples_train)  # for a gaussian, interquartile range is 1.349σ
    num_samples_val = x_val.shape[0]
    new_features_val = rng.multivariate_normal(mean=np.mean(x_train, axis=0)[cols_to_imitate],
                                               cov=np.diag(((np.quantile(x_train, 0.75, axis=0) - np.quantile(x_train,
                                                                                                              0.25,
                                                                                                              axis=0))[
                                                                cols_to_imitate] / 1.349) ** 2),
                                               size=num_samples_val)
    num_samples_test = x_test.shape[0]
    new_features_test = rng.multivariate_normal(mean=np.mean(x_train, axis=0)[cols_to_imitate],
                                                cov=np.diag(((np.quantile(x_train, 0.75, axis=0) - np.quantile(x_train,
                                                                                                               0.25,
                                                                                                               axis=0))[
                                                                 cols_to_imitate] / 1.349) ** 2),
                                                size=num_samples_test)

    return np.concatenate((x_train, new_features_train), axis=1), np.concatenate((x_val, new_features_val),
                                                                                 axis=1), np.concatenate(
        (x_test, new_features_test), axis=1), y_train, y_val, y_test


def gaussienize(x_train, x_val, x_test, y_train, y_val, y_test, type="standard", rng=None):
    """
    Gaussienize the data
    :param x: data to transform
    :param type: {"standard","robust", "quantile", "power", "quantile_uniform"}
    :return: the transformed data
    """
    print("Gaussienizing")
    if type == "identity":
        return x_train, x_val, x_test, y_train, y_val, y_test
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
    x_val = t.transform(x_val)
    x_test = t.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test


def cluster_1d(x, y, type="kmeans", rng=None, **kwargs):
    if type == "kmeans":
        for i in range(x.shape[1]):
            x[:, i] = KMeans(random_state=rng, **kwargs).fit_predict(x[:, i].reshape(-1, 1))
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)

    else:
        raise ValueError("Not implemented yet")

    return x, y


def remove_useless_features(x_train, x_val, x_test, y_train, y_val, y_test, max_rel_decrease=0.01, n_iter=3, rng=None):
    # TODO test
    if max_rel_decrease == 0:
        return x_train, x_val, x_test, y_train, y_val, y_test
    print("Removing useless_features...")
    rf = RandomForestClassifier(random_state=rng)  # used to compute features importance
    gbt = GradientBoostingClassifier(random_state=rng)  # used to compute accuracy decrease
    rf.fit(x_train, y_train)
    sorted_features = np.argsort(rf.feature_importances_)
    score_full_list = []
    for i in range(n_iter):
        x_train_bis, x_test_bis, y_train_bis, y_test_bis = train_test_split(x_train, y_train, test_size=0.5,
                                                                            random_state=rng)
        gbt.fit(x_train_bis, y_train_bis)
        score_full = gbt.score(x_test_bis, y_test_bis)
        score_full_list.append(score_full)
    i = 0
    features_to_remove = []
    for feature in sorted_features[:-1]:  # in order of increasing importance
        features_to_remove.append(feature)
        i += 1
        x_train_new = np.delete(x_train, features_to_remove, axis=1)
        score_new_list = []
        for j in range(n_iter):
            x_train_bis, x_test_bis, y_train_bis, y_test_bis = train_test_split(x_train_new, y_train, test_size=0.5,
                                                                                random_state=rng)
            gbt.fit(x_train_bis, y_train_bis)
            score_new = gbt.score(x_test_bis, y_test_bis)
            score_new_list.append(score_new)

        if (np.mean(score_full) - np.mean(score_new)) / np.mean(score_full) > max_rel_decrease:
            features_to_remove.pop()
            break

    return np.delete(x_train, features_to_remove, axis=1), np.delete(x_val, features_to_remove, axis=1), np.delete(
        x_test, features_to_remove, axis=1), y_train, y_val, y_test


def select_features_rf(x_train, x_val, x_test, y_train, y_val, y_test, rng, num_features=None, importance_cutoff=None,
                       return_features=False):
    assert (num_features is None) + (importance_cutoff is None) == 1  # xor
    if isinstance(num_features, float):
        num_features = int(num_features * x_train.shape[1])
    rf = RandomForestClassifier(random_state=rng)
    rf.fit(x_train, y_train)
    if importance_cutoff is not None:
        num_features = max(1, np.sum(rf.feature_importances_ > importance_cutoff))  # At least one feature

    x_train = x_train[:, np.argsort(rf.feature_importances_)[-num_features:]]
    x_val = x_val[:, np.argsort(rf.feature_importances_)[-num_features:]]
    x_test = x_test[:, np.argsort(rf.feature_importances_)[-num_features:]]
    if not return_features:
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test, np.argsort(rf.feature_importances_)[-num_features:]


def remove_features_rf(x_train, x_val, x_test, y_train, y_val, y_test, rng, num_features_to_remove=None,
                       importance_cutoff=None,
                       keep_removed_features=False,
                       return_features=False, model_to_use="rf_c"):
    assert (num_features_to_remove is None) + (importance_cutoff is None) == 1  # xor
    assert x_train.shape[1] == x_test.shape[1]
    if isinstance(num_features_to_remove, float):
        num_features_to_remove = int(num_features_to_remove * x_train.shape[1])
    model = model_keyword_dic[model_to_use](random_state=rng)
    model.fit(x_train, y_train)
    if importance_cutoff is not None:
        num_features_to_remove = min(x_train.shape[1] - 1,
                                     np.sum(model.feature_importances_ < importance_cutoff))  # At least one feature
    features_to_keep = np.argsort(model.feature_importances_)[- (x_train.shape[1] - num_features_to_remove):]
    if keep_removed_features:  # for experiment purposes, keep the features we should have removed
        features_to_keep = [i for i in range(x_train.shape[1]) if i not in features_to_keep]
    x_train = x_train[:, features_to_keep]
    x_val = x_val[:, features_to_keep]
    x_test = x_test[:, features_to_keep]
    assert x_train.shape[1] == x_test.shape[1]
    if not return_features:
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test, np.argsort(model.feature_importances_)[
                                                               -(x_train.shape[1] - num_features_to_remove):]


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


def remove_high_frequency_from_train(x_train, x_val, x_test, y_train, y_val, y_test, rng=None, cov_mult=0.001,
                                     covariance_estimation="classic", classif=True):
    if cov_mult == 0:
        return x_train, x_val, x_test, y_train, y_val, y_test
    y_train_new = np.zeros(y_train.shape)
    # empirical_cov = np.cov(x_train, rowvar=False)
    # empirical_cov = MinCovDet(support_fraction=1.0,
    #                          assume_centered=True).fit(x_train).covariance_
    if covariance_estimation == "robust":
        cov_method = MinCovDet(support_fraction=None,
                               assume_centered=False)
    elif covariance_estimation == "classic":
        cov_method = EmpiricalCovariance()
    else:
        raise NotImplemented
    empirical_cov = cov_method.fit(x_train).covariance_
    print(np.diag(empirical_cov))
    empirical_cov = cov_mult * empirical_cov
    for i in range(x_train.shape[0]):
        try:
            gaussian_kernel = multivariate_normal(mean=x_train[i], cov=empirical_cov)
        except:
            assert covariance_estimation == "robust"
            print("Issue with robust covaraince estimation, going for classic empirical estimation")
            cov_method = EmpiricalCovariance()
            empirical_cov = cov_method.fit(x_train).covariance_
            gaussian_kernel = multivariate_normal(mean=x_train[i], cov=empirical_cov)

        gaussian_densities = gaussian_kernel.pdf(x_train)
        #print(sorted(gaussian_densities)[-10:][::-1] / np.sum(gaussian_densities))
        y_train_new[i] = np.dot(y_train, gaussian_densities) / np.sum(gaussian_densities)
        #y_train_new[i] = (- y_train[i] * gaussian_densities[i] + np.dot(y_train, gaussian_densities)) / (np.sum(gaussian_densities) - gaussian_densities[i])
    if classif:
        y_train_new = (y_train_new > 0.5).astype(int)
        print(np.unique(y_train_new, return_counts=True))
    return x_train, x_val, x_test, y_train_new, y_val, y_test
