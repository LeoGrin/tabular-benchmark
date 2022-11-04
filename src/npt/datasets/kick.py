from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from npt.datasets.base import BaseDataset


class KickDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        The challenge of this Kaggle competition is to predict if the car
        purchased at the Auction is a Kick (bad buy).

        Accessed through OpenML
        https://www.openml.org/d/41162

        72983 rows
        32 features, 1 target column.

        Binary classification.

        Target in last column.

            Features                      n_unique   encode as
         0  PurchDate                            517          NUM
         1  Auction                              3            CAT
         2  VehYear                              10           NUM
         3  VehicleAge                           10           NUM
         4  Make                                 33           CAT
         5  Model                                1063         CAT
         6  Trim                                 134          CAT
         7  SubModel                             863          CAT
         8  Color                                16           CAT
         9  Transmission                         3            CAT
         10 WheelTypeID                          4            CAT
         11 WheelType                            3            CAT
         12 VehOdo                               39947        NUM
         13 Nationality                          4            CAT
         14 Size                                 12           CAT
         15 TopThreeAmericanName                 4            CAT
         16 MMRAcquisitionAuctionAveragePrice    10342        NUM
         17 MMRAcquisitionAuctionCleanPrice      11379        NUM
         18 MMRAcquisitionRetailAveragePrice     12725        NUM
         19 MMRAcquisitonRetailCleanPrice        13456        NUM
         20 MMRCurrentAuctionAveragePrice        10315        NUM
         21 MMRCurrentAuctionCleanPrice          11265        NUM
         22 MMRCurrentRetailAveragePrice	     12493        NUM
         23 MMRCurrentRetailCleanPrice           13192        NUM
         24 PRIMEUNIT                            2            CAT
         25 AUCGUART                             2            CAT
         26 BYRNO                                74           CAT
         27 VNZIP1                               153          CAT
         28 VNST                                 37           CAT
         29 VehBCost                             2010         NUM
         30 IsOnlineSale                         2            CAT
         31 WarrantyCost                         281          NUM
        """

        # Load data from https://www.openml.org/d/4353
        data_home = Path(self.c.data_path) / self.c.data_set
        x, y = fetch_openml(
            'kick',
            version=1, return_X_y=True, data_home=data_home)

        if isinstance(x, np.ndarray):
            pass
        elif isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        x = np.concatenate((x, np.expand_dims(y, -1)), axis=1)
        print(x.shape)

        self.data_table = x
        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        # Target col is the last feature
        self.num_target_cols = []
        self.cat_target_cols = [x.shape[1] - 1]

        self.num_features = [
            0, 2, 3, 12, 16, 17, 18, 19, 20, 21, 22, 23, 29, 31]
        self.cat_features = [
            1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 24, 25, 26, 27, 28, 30, 32]

        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.data_table, self.missing_matrix = self.impute_missing_entries(
            cat_features=self.cat_features, data_table=self.data_table,
            missing_matrix=self.missing_matrix)

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['openml']
