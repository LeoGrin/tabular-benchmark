from pathlib import Path

import numpy as np
import pandas as pd

from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download


class BreastCancerClassificationDataset(BaseDataset):
    def __init__(self, c):
        super(BreastCancerClassificationDataset, self).__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = load_and_preprocess_breast_cancer_dataset(
            self.c)

        # For breast cancer, target index is the first column
        self.num_target_cols = []
        self.cat_target_cols = [0]

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['wdbc.data']

        # overwrite missing
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0


def load_and_preprocess_breast_cancer_dataset(c):
    """Class imbalance is [357, 212]."""
    path = Path(c.data_path) / c.data_set
    data_name = 'wdbc.data'

    file = path / data_name

    if not file.is_file():
        # download if does not exist
        url = (
            'https://archive.ics.uci.edu/ml/'
            + 'machine-learning-databases/'
            + 'breast-cancer-wisconsin/'
            + data_name)

        download(file, url)

    # Read dataset
    data_table = pd.read_csv(file, header=None).to_numpy()

    # Drop id col
    data_table = data_table[:, 1:]

    N = data_table.shape[0]
    D = data_table.shape[1]

    if c.exp_smoke_test:
        print('Running smoke test -- building simple breast cancer dataset.')
        dm = data_table[data_table[:, 0] == 'M'][:8, :5]
        db = data_table[data_table[:, 0] == 'B'][:8, :5]
        data_table = np.concatenate([dm, db], 0)
        N = data_table.shape[0]
        D = data_table.shape[1]

        # Speculate some spurious missing features
        missing_matrix = np.zeros((N, D))
        missing_matrix[0, 1] = 1
        missing_matrix[2, 2] = 1
        missing_matrix = missing_matrix.astype(dtype=np.bool_)
    else:
        missing_matrix = np.zeros((N, D))
        missing_matrix = missing_matrix.astype(dtype=np.bool_)

    cat_features = [0]
    num_features = list(range(1, D))
    return data_table, N, D, cat_features, num_features, missing_matrix


class BreastCancerDebugClassificationDataset(BaseDataset):
    """For debugging row interactions. Add two columns for index tracking."""
    def __init__(self, c):
        super(BreastCancerClassificationDataset, self).__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        raise
        # need to augment table and features and and and
        # (to contain the index rows!! can already write index rows as long
        # as permutation is random!)

        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = load_and_preprocess_breast_cancer_dataset(
            self.c)

        # For breast cancer, target index is the first column
        self.num_target_cols = []
        self.cat_target_cols = [0]

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['wdbc.data']

        # overwrite missing
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0
