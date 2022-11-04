from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

from npt.datasets.base import BaseDataset


class YachtDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Regression dataset.

        Target in last column.
        308 rows.
        6 attributes.
        1 target (Residuary.resistance) (256 unique numbers)

        Features                    n_unique        encode as
        Logitudinal.position         5              CAT
        Prismatic.coefficient       10              CAT
        Length.displacement.ratio    8              CAT
        Beam.draught.ratio          17              CAT
        Length.beam.ratio           10              CAT
        Froude.number'              14              NUM


        Std of Target Col 15.135858907655322.
        """

        # Load data from https://www.openml.org/d/554
        data_home = Path(self.c.data_path) / self.c.data_set
        x, y = fetch_openml(
            'yacht_hydrodynamics',
            version=1, return_X_y=True, data_home=data_home)

        self.data_table = np.concatenate([x, y[:, np.newaxis]], 1)

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        # Target col is the last feature
        self.num_target_cols = [self.D - 1]
        self.cat_target_cols = []

        self.num_features = [self.D - 1]
        self.cat_features = list(range(0, self.D - 1))

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['openml']
