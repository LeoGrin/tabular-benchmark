import numpy as np


class NumEncoder(object):
    """
        cat_col: list of column names of categorical data
        num_col: list of column names of numerical data
    """

    def __init__(self, cat_col, num_col):

        self.cat_col = cat_col
        self.num_col = num_col

        self.means = []
        self.stds = []
        self.saved_sums = {}
        self.max_len = {}

    def fit_transform(self, df):

        print("Preprocess data for GBDT2NN...")

        # Preprocess the numerical data
        rows_num = self.min_max_scale(df.astype('float'))

        # Manual binary encoding
        if self.cat_col:
            rows_cat = self.binary_encoding(df)
            return np.concatenate([rows_num, rows_cat], axis=1)
        else:
            return rows_num

    def transform(self, df):

        # Preprocess the numerical data
        rows_num = self.min_max_scale(df.astype('float'), self.means, self.stds)

        # Manual binary encoding
        if self.cat_col:
            rows_cat = self.binary_encoding(df, self.max_len)
            return np.concatenate([rows_num, rows_cat], axis=1)
        else:
            return rows_num

    def refit_transform(self, df):

        # Update Means
        for item in self.num_col:
            self.saved_sums[item]['sum'] += df[item].sum()
            self.saved_sums[item]['cnt'] += df[item].shape[0]

        return self.transform(df)

    def binary_encoding(self, df, saved_bit_len=None):

        # print('Manual binary encode of categorical data')

        rows = []

        for item in self.cat_col:

            # Get all values from column
            # feats = df[item].values.astype(np.uint8).reshape((-1, 1))
            feats = df[:, item].astype(np.uint8).reshape((-1, 1))

            # Compute the needed bit length based on the max size of the values

            if saved_bit_len is None:
                bit_len = len(bin(df[:, item].astype(np.uint8).max())) - 2
                self.max_len[item] = bit_len
            else:
                bit_len = saved_bit_len[item]

            # change decimal to binary representation
            res = np.unpackbits(feats, axis=1, count=bit_len, bitorder='little')

            # append to all rows
            # rows = np.concatenate([rows,res],axis=1)
            rows.append(res)

        return np.concatenate(rows, axis=1)

    def min_max_scale(self, df, mean=None, std=None):
        # print('Min Max Scaling of numerical data')
        rows = df[:, self.num_col] #.to_numpy()

        if mean is None:
            mean = np.mean(rows, axis=0)
            self.means = mean

        if std is None:
            std = np.std(rows, axis=0)
            self.stds = std

        rows = (rows - mean) / (std + 1e-5)
        return rows
