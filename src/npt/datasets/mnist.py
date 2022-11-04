from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

from npt.datasets.base import BaseDataset


class MNISTDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=-10000)
        self.c = c

    def load(self):
        """
        Classification dataset.

        Target in last column.
        70 000 rows.
        784 attributes.
        1 class (10 class labels)

        Class imbalance: Not really.
        array([6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958])

        """

        # Load data from https://www.openml.org/d/554
        data_home = Path(self.c.data_path) / self.c.data_set
        x, y = fetch_openml(
            'mnist_784', version=1, return_X_y=True, data_home=data_home)

        self.data_table = np.hstack((x, np.expand_dims(y, -1)))

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]
        self.cat_features = [self.D-1]
        self.num_features = list(range(0, self.D-1))

        # Target col is the last feature
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['openml']
