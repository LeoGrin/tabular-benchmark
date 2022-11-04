from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from npt.datasets.base import BaseDataset


class ConcreteDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Regression dataset.

        Target in last column.
        1030 rows.
        8 attributes.
        1 target (Residuary.resistance) (256 unique numbers)

            Features                      n_unique   encode as
         0  Cement                         278          NUM
         1  Blast Furnace Slag             185          NUM
         2  Fly Ash                        156          NUM
         3  Water                          195          NUM
         4  Superplasticizer               111          NUM
         5  Coarse Aggregate               284          NUM
         6  Fine Aggregate                 302          NUM
         7  Age                             14          NUM
         8  Concrete compressive strength  845          NUM

        Std of Target Col 16.697630409134263.
        """

        # Load data from https://www.openml.org/d/4353
        data_home = Path(self.c.data_path) / self.c.data_set
        x, _ = fetch_openml(
            'Concrete_data',
            version=1, return_X_y=True, data_home=data_home)

        if isinstance(x, np.ndarray):
            pass
        elif isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        self.data_table = x
        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        # Target col is the last feature
        self.num_target_cols = [self.D - 1]
        self.cat_target_cols = []

        self.num_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.cat_features = []

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['openml']