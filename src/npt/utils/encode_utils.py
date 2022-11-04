from collections import deque

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def construct_encoded_col(
        non_missing_col_filter, encoded_non_missing_col_values):
    """
    Construct encoded column. For missing values:
        If column is numeric, take the entry from original col.
        If column is categorical, take an empty numpy array of one-hot length.

    :param non_missing_col_filter:
        np.array[np.bool_], True in entries that are not missing
    :param encoded_non_missing_col_values:
        np.array, Encoded column for non-missing values
    :return: np.array, encoded_col
    """
    encoded_col = []
    encoded_queue = deque(encoded_non_missing_col_values)
    one_hot_length = encoded_non_missing_col_values.shape[1]

    for elem_was_encoded in non_missing_col_filter:
        if elem_was_encoded:
            encoded_col.append(encoded_queue.popleft())
        else:
            # Replaces missing elements with a one-hot of correct length for
            # cat variables, and a single value for num variables
            encoded_col.append(np.zeros(one_hot_length))

    return np.array(encoded_col)


def get_compute_statistics_and_non_missing_matrix(data_dict, c):
    missing_matrix = data_dict['missing_matrix']
    val_mask_matrix = data_dict['val_mask_matrix']
    test_mask_matrix = data_dict['test_mask_matrix']
    row_boundaries = data_dict['row_boundaries']

    # Matrix with a 1 entry for all elements at which we
    # should compute a statistic / encode over
    compute_statistics_matrix = (
            1 - missing_matrix - val_mask_matrix -
            test_mask_matrix).astype(np.bool_)

    # If production, don't compute statistics using val/test
    if not c.model_is_semi_supervised:
        compute_statistics_matrix[row_boundaries['train']:] = False

    # Matrix with a 1 entry for all non-missing elements (i.e. those
    # we should transform)
    non_missing_matrix = ~missing_matrix

    return compute_statistics_matrix, non_missing_matrix, missing_matrix


def encode_data(
        data_dict, compute_statistics_matrix, non_missing_matrix,
        missing_matrix, data_dtype, use_bert_masking, c):
    """
    :return:
    Unpacked from data_dict:
        :param data_table: np.array, 2D unencoded data array
        :param N: int, number of rows
        :param D: int, number of columns
        :param cat_features: List[int], column indices with cat features
        :param num_features: List[int], column indices with num features
    :param compute_statistics_matrix: np.array[bool], True entries in
        locations that should be used to compute statistics / fit encoder
    :param non_missing_matrix: np.array[bool], True entries in
        locations that are not missing from data (don't attempt to encode
        missing entries, which could be NaN or an arbitrary missing token)
    :param missing_matrix: np.array[bool], inverse of the above
    :param np dtype to use for all data arrays
    :param use_bert_masking, if False, do not add an extra column to keep track
        of masked and missing values
    :return: Tuple[encoded_data, input_feature_dims]
            encoded_dataset: List[np.array], encoded columns
            input_feature_dims: List[int], Size of encoding for each feature.
                Needed to initialise embedding weights in NPT.
    """
    data_table = data_dict['data_table']
    N = data_dict['N']
    D = data_dict['D']
    cat_features = data_dict['cat_features']
    num_features = data_dict['num_features']
    cat_target_cols = data_dict['cat_target_cols']

    encoded_dataset = []
    input_feature_dims = []

    standardisation = np.nan * np.ones((D, 2))
    tabnet_mode = (c.model_class == 'sklearn-baselines' and
                   c.sklearn_model == 'TabNet')

    # Extract just the sigmas in a JSON-serializable format
    # we use this as metadata for numerical columns to unstandardize them
    sigmas = []
    if tabnet_mode:
        cat_col_dims = []

    for col_index in range(D):
        # The column over which we compute statistics
        stat_filter = compute_statistics_matrix[:, col_index]
        stat_col = data_table[stat_filter, col_index].reshape(-1, 1)

        # Non-missing entries, which we transform
        non_missing_filter = non_missing_matrix[:, col_index]
        non_missing_col = data_table[
            non_missing_filter, col_index].reshape(-1, 1)

        # Fit on stat_col, transform non_missing_col
        is_cat = False
        if col_index in cat_features:
            is_cat = True
            if tabnet_mode and col_index not in cat_target_cols:
                # Use TabNet's label encoding
                # https://github.com/dreamquark-ai/tabnet/blob/develop/
                # forest_example.ipynb
                l_enc = LabelEncoder()
                encoded_col = np.expand_dims(
                    l_enc.fit_transform(non_missing_col), -1)
                num_classes = len(l_enc.classes_)
                cat_col_dims.append(num_classes)
            else:
                fitted_encoder = OneHotEncoder(sparse=False).fit(
                    non_missing_col)
                encoded_col = fitted_encoder.transform(
                    non_missing_col).astype(np.bool_)

            # Stand-in for a np.nan, but JSON-serializable
            sigmas.append(-1)

        elif col_index in num_features:
            fitted_encoder = StandardScaler().fit(stat_col)
            encoded_col = fitted_encoder.transform(non_missing_col)
            standardisation[col_index, 0] = fitted_encoder.mean_[0]
            standardisation[col_index, 1] = fitted_encoder.scale_[0]
            sigmas.append(fitted_encoder.scale_[0])
        else:
            raise NotImplementedError

        # Construct encoded column
        # (we have only encoded non-missing entries! need to fill in missing)
        encoded_col = construct_encoded_col(
            non_missing_col_filter=non_missing_filter,
            encoded_non_missing_col_values=encoded_col)

        if use_bert_masking:
            # Add mask tokens to numerical and categorical data
            # Each col is now shape Nx(H_j+1)
            encoded_col = np.hstack(
                [encoded_col, np.zeros((N, 1))])

            # Get missing indices to zero out values and set mask token
            # TODO: try randomly sampling for missing indices from a std normal
            missing_filter = missing_matrix[:, col_index]

            # Zero out all one-hots (or the single numerical val) for these entries
            encoded_col[missing_filter, :] = 0

            # Set their mask token to 1
            encoded_col[missing_filter, -1] = 1

        if not tabnet_mode:
            # If categorical column, convert to bool
            if is_cat:
                encoded_col = encoded_col.astype(np.bool_)
            else:
                encoded_col = encoded_col.astype(data_dtype)

        encoded_dataset.append(encoded_col)
        input_feature_dims.append(encoded_col.shape[1])

    if tabnet_mode:
        return (
            encoded_dataset, input_feature_dims, standardisation,
            sigmas, cat_col_dims)
    else:
        return encoded_dataset, input_feature_dims, standardisation, sigmas


def encode_data_dict(data_dict, c):
    # * TODO: need to vectorize for huge datasets
    # * TODO: (i.e. can't fit in CPU memory)
    compute_statistics_matrix, non_missing_matrix, missing_matrix = (
        get_compute_statistics_and_non_missing_matrix(data_dict, c))

    data_dtype = get_numpy_dtype(dtype_name=c.data_dtype)

    return encode_data(
        data_dict, compute_statistics_matrix, non_missing_matrix,
        missing_matrix, data_dtype, c.model_bert_augmentation, c)


def get_numpy_dtype(dtype_name):
    if dtype_name == 'float32':
        dtype = np.float32
    elif dtype_name == 'float64':
        dtype = np.float64
    else:
        raise NotImplementedError

    return dtype


def get_torch_dtype(dtype_name):
    if dtype_name == 'float32':
        dtype = torch.float32
    elif dtype_name == 'float64':
        dtype = torch.float64
    else:
        raise NotImplementedError

    return dtype


def get_torch_tensor_type(dtype_name):
    if dtype_name == 'float32':
        dtype = torch.FloatTensor
    elif dtype_name == 'float64':
        dtype = torch.DoubleTensor
    else:
        raise NotImplementedError

    return dtype


def torch_cast_to_dtype(obj, dtype_name):
    if dtype_name == 'float32':
        obj = obj.float()
    elif dtype_name == 'float64':
        obj = obj.double()
    elif dtype_name == 'long':
        obj = obj.long()
    else:
        raise NotImplementedError

    return obj
