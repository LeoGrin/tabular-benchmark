from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from npt.datasets.base import BaseDataset


class IncomeDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=-99762)
        self.c = c

    def load(self):
        """KDD Income Dataset

        Possibly used in VIME and TabNet.

        There are multiple datasets called income.
        https://archive.ics.uci.edu/ml/datasets/census+income
        https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
        The KDD One is significantly larger than the other one.

        We will take KDD one. Both TabNet and VIME are not super explicit about
        which dataset they use.
        TabNet cite Oza et al "Online Bagging and Boosting", which use the
        bigger one. So we will start with that.
        (But there is no full TabNet Code to confirm.)

        Binary classification.

        Target in last column.
        299.285 rows.
        42 attributes. Use get_num_cat_auto to assign.
        1 target
        """

        # Load data from https://www.openml.org/d/4535
        data_home = Path(self.c.data_path) / self.c.data_set
        data = fetch_openml('Census-income', version=1, data_home=data_home)

        # target in 'data'
        self.data_table = data['data']

        if isinstance(self.data_table, np.ndarray):
            pass
        elif isinstance(self.data_table, pd.DataFrame):
            self.data_table = self.data_table.to_numpy()

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        # Target col is the last feature
        # last column is target (V42)
        # (binary classification, if income > or < 50k)
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        self.num_features, self.cat_features = BaseDataset.get_num_cat_auto(
            self.data_table, cutoff=55)
        print('income num cat features')
        print(len(self.num_features))
        print(len(self.cat_features))

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['openml']