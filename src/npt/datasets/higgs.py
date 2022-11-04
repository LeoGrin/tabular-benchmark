from pathlib import Path

import numpy as np
import pandas as pd
import patoolib

from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download
from os import remove


class HiggsClassificationDataset(BaseDataset):
    def __init__(self, c):
        super(HiggsClassificationDataset, self).__init__(
            fixed_test_set_index=-500000)   # Test set: last 500,000 examples
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = load_and_preprocess_higgs_dataset(
            self.c)

        self.num_target_cols = []
        self.cat_target_cols = [0]  # Binary classification
        self.is_data_loaded = True
        self.tmp_file_names = ['HIGGS.csv']


def load_and_preprocess_higgs_dataset(c):
    """HIGGS dataset as used by NODE.

    Binary classification.
    First column is categorical target column,
    all remaining 28 columns are continuous features.
    11.000.000 rows in total.
    The last 500,000 rows are commonly used as a test set.
    Separate training and test set.

    No class imbalance (array([0., 1.]), array([5170877, 5829123])).
    """
    path = Path(c.data_path) / c.data_set
    data_name = 'HIGGS.csv'
    file = path / data_name

    # For breast cancer, target index is the first column
    if not file.is_file():
        # download if does not exist
        download_name = 'HIGGS.csv.gz'
        url = (
                'https://archive.ics.uci.edu/ml/'
                + 'machine-learning-databases/00280/'
                + download_name
        )
        download_file = path / download_name
        download(download_file, url)

        # Higgs comes compressed.
        print('Decompressing...')
        patoolib.extract_archive(str(download_file), outdir=str(path))
        print('... done.')

        # Delete the compressed file (Higgs is very large)
        remove(download_file)
        print(f'Removed compressed file {download_name}.')

    data_table = pd.read_csv(file, header=None).to_numpy()
    N, D = data_table.shape
    cat_features = [0]
    num_features = list(range(1, D))
    missing_matrix = np.zeros((N, D), dtype=np.bool_)

    return data_table, N, D, cat_features, num_features, missing_matrix
