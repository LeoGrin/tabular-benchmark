from pathlib import Path

import numpy as np
import pandas as pd
import patoolib

from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download


class ForestCoverClassificationDataset(BaseDataset):
    def __init__(self, c):
        super(ForestCoverClassificationDataset, self).__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = load_and_preprocess_forest_cover_dataset(
            self.c)

        # Target col is the last feature -- multiclass classification
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        self.tmp_file_or_dir_names = ['covtype.data', 'covtype.data.gz']
        self.is_data_loaded = True


def load_and_preprocess_forest_cover_dataset(c):
    """ForestCoverDataset.

    Used in TabNet.

    Multi-class classification.
    Target in last column. (7 different categories of forest cover.)
    581,012 rows.
    Each row has 54 features (55th column is the target).

    Feature types:
        10 continuous features, 4 binary "wilderness area" features,
        40 binary "soil type" variables.

    Classical usage:
        first 11,340 records used for training data subset
        next 3,780 records used for validation data subset
        last 565,892 records used for testing data subset

    WE DON'T DO THE ABOVE, following the TabNet and XGBoost baselines
    Just do (0.8, 0.2) (train, test) split.

    Class imbalance: Yes.
    [211840, 283301,  35754,   2747,   9493,  17367,  20510]
    Guessing performance is 0.488 percent accuracy.
    Getting the two most frequent classes gives 0.729 percent accuracy.
    Top three most frequent gets 0.914.

    """

    path = Path(c.data_path) / c.data_set
    data_name = 'covtype.data'
    file = path / data_name

    if not file.is_file():
        # download if does not exist
        download_name = 'covtype.data.gz'
        url = (
                'https://archive.ics.uci.edu/ml/'
                + 'machine-learning-databases/covtype/'
                + download_name
        )
        download_file = path / download_name
        download(download_file, url)
        # Forest cover comes compressed.
        patoolib.extract_archive(str(download_file), outdir=str(path))

    data_table = pd.read_csv(file, header=None).to_numpy()

    # return
    if c.exp_smoke_test:
        print(
            'Running smoke test -- building simple forest cover dataset.')
        class_datasets = []
        for class_type in [1, 2, 3, 4, 5, 6, 7]:
            class_datasets.append([data_table[
                data_table[:, -1] == class_type][0]])

        data_table = np.concatenate(class_datasets, axis=0)

    N = data_table.shape[0]
    D = data_table.shape[1]
    num_features = list(range(10))
    cat_features = list(range(10, D))

    # TODO: add missing entries to sanity check
    missing_matrix = np.zeros((N, D), dtype=np.bool_)

    return data_table, N, D, cat_features, num_features, missing_matrix
