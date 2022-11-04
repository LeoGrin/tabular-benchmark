import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseDataset(ABC):
    """Abstract base dataset class.

    Requires subclasses to override the following:
        load, which should set all attributes below that are None
        which will be needed by the dataset (e.g. fixed_split_indices
        need only be set by a dataset that has a fully specified
        train, val, and test set for comparability to prior approaches).
    """
    def __init__(
            self, fixed_test_set_index):
        """
        Args:
            fixed_test_set_index: int, if specified, the dataset has a
                fixed test set starting at this index. Needed for
                comparability to other methods.
        """
        self.fixed_test_set_index = fixed_test_set_index
        self.c = None
        self.data_table = None
        self.missing_matrix = None
        self.N = None
        self.D = None
        self.cat_features = None
        self.num_features = None
        self.cat_target_cols = None
        self.num_target_cols = None
        self.auroc_setting = None
        self.is_data_loaded = False
        self.tmp_file_or_dir_names = []  # Deleted if c.clear_tmp_files=True

        # fixed_split_indices: Dict[str, np.array], a fully specified
        #   mapping from the dataset mode key (train, val, or test)
        #   to a np.array containing the indices for the respective
        #   mode.
        self.fixed_split_indices = None

    def get_data_dict(self, force_disable_auroc=None):
        if not self.is_data_loaded:
            self.load()

        self.auroc_setting = self.use_auroc(force_disable_auroc)

        # # # For some datasets, we should immediately delete temporary files
        # # # e.g. Higgs: zipped and unzipped file = 16 GB, CV split is 3 GB
        if self.c.data_clear_tmp_files:
            print('\nClearing tmp files.')
            path = Path(self.c.data_path) / self.c.data_set
            for file_or_dir in self.tmp_file_or_dir_names:
                file_dir_path = path / file_or_dir

                # Could be both file and a path!
                if os.path.isfile(file_dir_path):
                    os.remove(file_dir_path)
                    print(f'Removed file {file_or_dir}.')
                if os.path.isdir(file_dir_path):
                    try:
                        shutil.rmtree(file_dir_path)
                        print(f'Removed dir {file_or_dir}.')
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

        return self.__dict__

    @abstractmethod
    def load(self):
        pass

    def use_auroc(self, force_disable=None):
        """
        Disable AUROC metric:
            (i)  if we do not have a single categorical target column,
            (ii) if the single categorical target column is multiclass.
        """
        if not self.is_data_loaded:
            self.load()

        disable = 'Disabling AUROC metric.'

        if force_disable:
            print(disable)
            return False

        if not self.c.metrics_auroc:
            print(disable)
            print("As per config argument 'metrics_auroc'.")
            return False

        num_target_cols, cat_target_cols = (
            self.num_target_cols, self.cat_target_cols)
        n_cat_target_cols = len(cat_target_cols)

        if n_cat_target_cols != 1:
            print(disable)
            print(
                f'\tBecause dataset has {n_cat_target_cols} =/= 1 '
                f'categorical target columns.')
            if n_cat_target_cols > 1:
                print(
                    '\tNote that we have not decided how we want to handle '
                    'AUROC among multiple categorical target columns.')
            return False
        elif num_target_cols:
            print(disable)
            print(
                '\tBecause dataset has a nonzero count of '
                'numerical target columns.')
            return False
        else:
            auroc_col = cat_target_cols[0]
            if n_classes := len(np.unique(self.data_table[:, auroc_col])) > 2:
                print(disable)
                print(f'\tBecause AUROC does not (in the current implem.) '
                      f'support multiclass ({n_classes}) classification.')
                return False

        return True

    @staticmethod
    def get_num_cat_auto(data, cutoff):
        """Interpret all columns with < "cutoff" values as categorical."""
        D = data.shape[1]
        cols = np.arange(0, D)
        unique_vals = np.array([np.unique(data[:, col]).size for col in cols])

        num_feats = cols[unique_vals > cutoff]
        cat_feats = cols[unique_vals <= cutoff]

        assert np.intersect1d(cat_feats, num_feats).size == 0
        assert np.union1d(cat_feats, num_feats).size == D

        # we dump to json later, it will crie if not python dtypes
        num_feats = [int(i) for i in num_feats]
        cat_feats = [int(i) for i in cat_feats]

        return num_feats, cat_feats

    @staticmethod
    def impute_missing_entries(cat_features, data_table, missing_matrix):
        """
        Fill categorical missing entries with ?
        and numerical entries with the mean of the column.
        """
        for col in range(data_table.shape[1]):
            # Get missing value locations
            curr_col = data_table[:, col]

            if curr_col.dtype == np.object_:
                col_missing = np.array(
                    [True if str(n) == "nan" else False for n in curr_col])
            else:
                col_missing = np.isnan(data_table[:, col])

            # There are missing values
            if col_missing.sum() > 0:
                # Set in missing matrix (used to avoid using data augmentation
                # or predicting on those values
                missing_matrix[:, col] = col_missing

            if col in cat_features:
                missing_impute_val = '?'
            else:
                missing_impute_val = np.mean(
                    data_table[~col_missing, col])

            data_table[:, col] = np.array([
                missing_impute_val if col_missing[i] else data_table[i, col]
                for i in range(data_table.shape[0])])

        n_missing_values = missing_matrix.sum()
        print(f'Detected {n_missing_values} missing values in dataset.')

        return data_table, missing_matrix

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

        return missing_complete
