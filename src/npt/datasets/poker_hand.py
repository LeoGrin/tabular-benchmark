from pathlib import Path

import numpy as np
import pandas as pd

from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download


class PokerHandDataset(BaseDataset):
    def __init__(self, c):
        super(PokerHandDataset, self).__init__(
            fixed_test_set_index=None)  # Set when load is called
        self.c = c

    def load(self):
        """Poker Hand data set as used by TabNet.

        10-fold classification. (What kind of Poker Hand?)
        Target in last column.
        1025010 rows.
        Each row has 10 features which describe the poker hand.
        5 are numerical (the rank), 5 categorical (the suit).
        Last column is the label.
        Separate training and test set.

        This dataset has extreme class imbalance
        [513702, 433097, 48828, 21634, 3978, 2050, 1460, 236, 17, 8]
        such that guessing performance is 0.5.

        This also means that TabNet can get 99% performance by just getting
        the first 4 predictions right.

        # NOTE: UCI lists 'Ranks of Cards' as numerical feature
        # NOTE: but it seems categorical to me.

        """
        path = Path(self.c.data_path) / self.c.data_set

        data_names = ['poker-hand-training-true.data',
                      'poker-hand-testing.data']
        files = [path / data_name for data_name in data_names]

        files_exist = [file.is_file() for file in files]

        if not all(files_exist):
            url = (
                    'https://archive.ics.uci.edu/ml/'
                    + 'machine-learning-databases/poker/'
            )

            urls = [url + data_name for data_name in data_names]

            download(files, urls)

        data_tables = [
            pd.read_csv(file, header=None).to_numpy() for file in files]
        self.fixed_test_set_index = -data_tables[1].shape[0]
        self.data_table = np.concatenate(data_tables, 0)

        self.N, self.D = self.data_table.shape

        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        # It turns out that not all features in Poker Hands are categorical --
        # i.e., we can encode the rank of a card with a numerical variable.
        # self.cat_features = list(range(self.D))
        # self.num_features = []

        self.cat_features = [0, 2, 4, 6, 8, 10]
        self.num_features = [1, 3, 5, 7, 9]
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = data_names
