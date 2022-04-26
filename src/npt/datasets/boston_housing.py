import numpy as np
from sklearn.datasets import load_boston

from npt.datasets.base import BaseDataset


class BostonHousingDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Regression dataset.

        Target in last column.
        506 rows.
        13 attributes.

        Feature types (Copied from sklearn description):
             idx name    type num unique
             0 - CRIM     NUM  504  per capita crime rate by town
             1 - ZN       NUM   26  proportion of residential land zoned for
                                    lots over 25,000 sq.ft. (only has 26 unique
                                    values)
             2 - INDUS    NUM   76  proportion of non-retail business acres per
                                    town
             3 - CHAS     CAT    2  Charles River dummy variable (= 1 if tract
                                    bounds river; 0 otherwise)
             4 - NOX      NUM   81  nitric oxides concentration (parts per 10
                                    million)
             5 - RM       NUM  446  average number of rooms per dwelling
             6 - AGE      NUM  356  proportion of owner-occupied units built
                                    prior to 1940
             7 - DIS      NUM  412  weighted distances to five Boston
                                    employment centres
             8 - RAD      CAT    9  index of accessibility to radial highways
             9 - TAX      NUM   66  full-value property-tax rate per $10,000
            10 - PTRATIO  NUM   46  pupil-teacher ratio by town
            11 - B        NUM  357  1000(Bk - 0.63)^2 where Bk is the
                                    proportion of black individuals by town
            12 - LSTAT    NUM  455  % lower status of the population
        (T) 13 - MEDV     NUM  Median value of owner-occupied homes in $1000's

        Mean, std value of target column: 22.532806324110677, 9.188011545278203

        --> Just guessing the mean value on standardized data will always
        give you an MSE of 1.

        """

        x, y = load_boston(return_X_y=True)

        self.data_table = np.concatenate([x, y[:, np.newaxis]], 1)

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]
        self.cat_features = [3, 8]
        self.num_features = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13]

        # Target col is the last feature (numerical, "median housing value")
        self.num_target_cols = [self.D - 1]
        self.cat_target_cols = []

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.is_data_loaded = True

        # No tmp files left by this dwnld method
        self.tmp_file_or_dir_names = []
