"""Generic data utils invoked by dataset loaders."""

import numpy as np
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer


# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('k-bin-discretize', KBinsDiscretizer(n_bins=10, strategy='quantile'))])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


"""Preprocessing Functions: Classification and Regression Datasets."""


def get_dense_from_dok(dok_matrix):
    return np.array(
        [list(key) for key in dok_matrix.keys()])


def get_matrix_from_rows(rows, cols, N, D):
    """
    Constructs dense matrix with True in all locations where a label is.

    Labels occur in the specified rows, for each col in cols.
    """
    matrix = np.zeros((N, D), dtype=np.bool_)
    for col in cols:
        matrix[rows, col] = True

    return matrix


def get_entries_from_rows(rows, col, D):
    """
    Given list of rows return list of [rows, col], where col is repeated over
    elements of list.
    """
    if type(col) != int:
        raise NotImplementedError

    N = len(rows)
    entries = np.stack([rows, col * np.ones(N)], axis=-1)
    return entries


def indices_to_matrix_entries(indices, n_cols):
    """Convert list of 1D indices to 2D matrix indices.

    1D indices enumerate all positions in matrix, while 2D indices enumerate
    the rows and columns separately.
    Input:
        indices (np.array, N*n_cols): List of 1D indices.
        n_cols (int): Number of columns in target matrix.
    Returns:
        matrix_entries (np.array, (N, n_cols)): Matrix entries. Equal to a
            sparse representation.

    """
    if type(indices) == list:
        indices = np.array(indices)
    rows = indices // n_cols
    cols = indices % n_cols
    matrix_entries = np.stack([rows, cols], 1)
    return matrix_entries


def entries_to_dense(entries, N, D):
    """Convert array of binary masking entries to dense matrix.

    Input:
        entries (np.array, 2xM): List of sparse positions.
        N: Number of rows.
        D: Number of cols.

    Returns:
        dense_matrix (np.array, NxD): Dense matrix with 1 for all entries,
            else 0.
    """
    # check for empty
    if entries.size == 0:
        return np.zeros((N, D))

    data = np.ones(entries.shape[0])
    sparse_matrix = sparse.csr_matrix(
        (data, (entries[:, 0], entries[:, 1])), shape=(N, D), dtype=np.bool_)
    dense_matrix = sparse_matrix.toarray().astype(dtype=np.bool_)

    assert set(np.where(dense_matrix == 1)[0]) == set(entries[:, 0])
    assert set(np.where(dense_matrix == 1)[1]) == set(entries[:, 1])

    return dense_matrix
