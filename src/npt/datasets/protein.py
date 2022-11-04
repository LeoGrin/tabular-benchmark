from pathlib import Path

import numpy as np
import pandas as pd

from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download


def load_protein(c, data_name):
    """Protein Dataset

    Used in Gal et al., 'Dropout as Bayesian Approximation'.

    Physicochemical Properties of Protein Tertiary Structure Data Set

    Regression Dataset
    Number of Rows 45730
    Number of Attributes 9

    RMSD-Size of the residue.
    F1 - Total surface area.
    F2 - Non polar exposed area.
    F3 - Fractional area of exposed non polar residue.
    F4 - Fractional area of exposed non polar part of residue.
    F5 - Molecular mass weighted exposed area.
    F6 - Average deviation from standard exposed area of residue.
    F7 - Euclidian distance.
    F8 - Secondary structure penalty.
    F9 - Spatial Distribution constraints (N,K Value).

    There may be a fixed test set as suggested by 'more-documentation.
    names' but it does not seem like Hernandez-Lobato et al. (whose setup
    Gal et al. repeat), respect that.

    https://www.kaggle.com/c/pcon-ml seems to suggest that RMSD is target.

    Target Col has std of 6.118244779017878.
    """
    path = Path(c.data_path) / c.data_set

    file = path / data_name

    if not file.is_file():
        # download if does not exist
        url = (
            'https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/00265/'
            + data_name
            )
        download_file = path / data_name
        download(download_file, url)

    return pd.read_csv(file).to_numpy()


class ProteinDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)

        self.c = c

    def load(self):
        data_name = 'CASP.csv'
        self.data_table = load_protein(self.c, data_name)
        self.N, self.D = self.data_table.shape
        self.num_target_cols = [0]
        self.cat_target_cols = []

        # have checked this with get_num_cat_auto as well
        self.cat_features = []
        self.num_features = list(range(0, self.D))

        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            # self.data_table[self.missing_matrix] = 0

        else:
            self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = [data_name]

    def make_missing(self, p):
        N = self.N
        D = self.D

        # drawn random indices (excluding the target columns)
        target_cols = self.num_target_cols + self.cat_target_cols
        D_miss = D - len(target_cols)

        missing = np.zeros((N * D_miss), dtype=np.bool_)

        # draw random indices at which to set True do
        idxs = np.random.choice(
            a=range(0, N * D_miss), size=int(p * N * D_miss), replace=False)

        # set missing to true at these indices
        missing[idxs] = True

        assert missing.sum() == int(p * N * D_miss)

        # reshape to original shape
        missing = missing.reshape(N, D_miss)

        # add back target columns
        missing_complete = missing

        for col in target_cols:
            missing_complete = np.concatenate(
                [missing_complete[:, :col],
                 np.zeros((N, 1), dtype=np.bool_),
                 missing_complete[:, col:]],
                axis=1
            )

        if len(target_cols) > 1:
            raise NotImplementedError(
                'Missing matrix generation should work for multiple '
                'target cols as well, but this has not been tested. '
                'Please test first.')

        print(missing_complete.shape)
        return missing_complete
