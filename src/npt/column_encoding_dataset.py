import json
import os
import pickle

import numpy as np
import torch

from npt.batch_dataset import NPTBatchDataset
from npt.datasets.base import BaseDataset
from npt.datasets.boston_housing import BostonHousingDataset
from npt.datasets.breast_cancer import (
    BreastCancerClassificationDataset, BreastCancerDebugClassificationDataset)
from npt.datasets.cifar10 import CIFAR10Dataset
from npt.datasets.concrete import ConcreteDataset
from npt.datasets.debug import DebugDataset
from npt.datasets.forest_cover import ForestCoverClassificationDataset
from npt.datasets.higgs import HiggsClassificationDataset
from npt.datasets.image_utils import load_image_dataloaders
from npt.datasets.income import IncomeDataset
from npt.datasets.kick import KickDataset
from npt.datasets.mnist import MNISTDataset
from npt.datasets.poker_hand import PokerHandDataset
from npt.datasets.protein import ProteinDataset
from npt.datasets.yacht import YachtDataset
from npt.utils.cv_utils import (
    get_class_reg_train_val_test_splits, get_n_cv_splits)
from npt.utils.encode_utils import encode_data_dict
from npt.utils.memory_utils import get_size
from npt.utils.preprocess_utils import (
    get_matrix_from_rows)

DATASET_NAME_TO_DATASET_MAP = {
    'breast-cancer': BreastCancerClassificationDataset,
    'poker-hand': PokerHandDataset,
    'higgs': HiggsClassificationDataset,
    'forest-cover': ForestCoverClassificationDataset,
    'boston-housing': BostonHousingDataset,
    'mnist': MNISTDataset,
    'yacht': YachtDataset,
    'concrete': ConcreteDataset,
    'income': IncomeDataset,
    'protein': ProteinDataset,
    'breast-cancer-debug': BreastCancerDebugClassificationDataset,
    'debug': DebugDataset,
    'cifar10': CIFAR10Dataset,
    'kick': KickDataset
}

# Preprocessed separately for data augmentation
IMAGE_DATASETS = ['cifar10']

TORCH_MASK_MATRICES = [
    'missing_matrix',
    'train_mask_matrix', 'val_mask_matrix', 'test_mask_matrix',
    # 'bert_mask_matrix'
]

METADATA_FIELDS = [
    'N', 'D', 'cat_features', 'num_features',
    'cat_target_cols', 'num_target_cols', 'input_feature_dims',
    'fixed_test_set_index', 'auroc_setting']


class ColumnEncodingDataset:
    """
    Dataset constructed from columns of various encoding sizes.

    Tuple of (row_independent_inference, mode) jointly determines
    batching strategy for NPT model.
    """
    def __init__(self, c, device=None):
        super(ColumnEncodingDataset).__init__()

        self.c = c
        self.device = c.exp_device if device is None else device

        # Together with mode determines batching strategy
        self.is_torch_model = self.get_model_details(self.c)
        self.mode = None
        self.valid_modes = ['train', 'val', 'test']

        # Retrieve dataset class and metadata
        try:
            self._dataset = DATASET_NAME_TO_DATASET_MAP[
                self.c.data_set](self.c)  # type: BaseDataset
        except KeyError:
            raise NotImplementedError(
                f'Have not implemented dataset {self.c.data_set}')

        # Retrieve pathing information
        self.cache_path, self.model_cache_path, self.n_cv_splits = (
            self.init_cache_path_and_splits())
        self.metadata_path = os.path.join(
            self.cache_path, 'dataset__metadata.json')
        # Generate dataset
        self.reset_cv_splits()

    # Allows reuse of the same dataset object, for e.g. multiple sklearn models
    # NOTE: does not reset metadata, which is same across cv splits
    def reset_cv_splits(self):
        self.dataset_gen = self.run_preprocessing_and_caching()
        self.curr_cv_split = -1
        self.cv_dataset = None

    def load_next_cv_split(self):
        self.curr_cv_split += 1
        if self.curr_cv_split > self.n_cv_splits:
            raise Exception(
                'Have loaded too many datasets for our n_cv_splits.')

        self.cv_dataset = next(self.dataset_gen)

    """Model and Mode Settings"""

    def get_model_details(self, c):
        """
        :return: is_torch_model
        """
        # Determine if model is expecting torch tensors
        return c.model_class == 'NPT'

    def set_mode(self, mode, epoch):
        assert mode in self.valid_modes

        if self.curr_cv_split == -1:
            raise Exception(
                'CV split dataset has not been loaded. '
                'Call dataset.load_next_cv_split')

        self.mode = mode

        if self.c.verbose:
            print(
                f'Loading {mode} batches for CV split '
                f'{self.curr_cv_split + 1}, epoch {epoch + 1}.')

        # Loads new batches in the CV dataset
        self.cv_dataset.set_mode(mode, epoch)

        if self.c.verbose:
            print('Successfully loaded batch.')

    def is_mode_set(self):
        return self.mode is not None

    """Preprocessing: Pathing"""

    def init_cache_path_and_splits(self):
        n_cv_splits = get_n_cv_splits(self.c)
        ssl_str = f'ssl__{self.c.model_is_semi_supervised}'
        cache_path = os.path.join(
            self.c.data_path, self.c.data_set, ssl_str,
            f'np_seed={self.c.np_seed}__n_cv_splits={n_cv_splits}'
            f'__exp_num_runs={self.c.exp_n_runs}')

        if self.c.model_checkpoint_key is not None:
            model_cache_path = os.path.join(
                cache_path, self.c.model_checkpoint_key)
        else:
            model_cache_path = cache_path

        if not os.path.exists(cache_path):
            try:
                os.makedirs(cache_path)
            except FileExistsError as e:
                print(e)

        if not os.path.exists(model_cache_path):
            try:
                os.makedirs(model_cache_path)
            except FileExistsError as e:
                print(e)

        return cache_path, model_cache_path, n_cv_splits

    def are_datasets_cached(self):
        if self.c.data_force_reload:
            # TODO: should rename to data_force_rebuild probably
            print('Forcing data rebuild and recache.')
            return False

        cached_dataset_filenames = [
            filename for filename in os.listdir(self.cache_path)
            if 'dataset' in filename]

        expected_dataset_filenames = sorted([
            f'dataset__split={cv_split}.pkl' for cv_split in range(
                min(self.n_cv_splits, self.c.exp_n_runs))] + [
                'dataset__metadata.json'])

        datasets_are_cached = (
            sorted(cached_dataset_filenames) ==
            sorted(expected_dataset_filenames))

        if datasets_are_cached:
            print('CV Splits for this dataset are cached. Loading from file.')

        return datasets_are_cached

    """Preprocessing: Load and Cache"""

    def load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def cache_metadata(self, data_dict):
        # Fields which are generic for each CV split
        metadata_dict = {
            key: data_dict[key]
            for key in METADATA_FIELDS}

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_dict, fp=f)

    def get_image_metadata(self, data_dict):
        return {
            key: data_dict[key]
            for key in METADATA_FIELDS}

    def load_datasets(self):
        for cv_split in range(min(self.n_cv_splits, self.c.exp_n_runs)):
            dataset_path = os.path.join(
                self.cache_path, f"dataset__split={cv_split}.pkl")
            with open(dataset_path, 'rb') as f:
                data_dict = pickle.load(file=f)

            if self.c.data_log_mem_usage:
                print(
                    f'Recursive size of dataset: '
                    f'~{get_size(data_dict)/(1024 * 1024 * 1024):.6f}'
                    f' GB')

            if self.is_torch_model:
                yield self.load_torch_dataset(data_dict)
            else:
                data_dict['data_arrs'] = data_dict['data_table']
                del data_dict['data_table']
                yield data_dict

    def load_torch_dataset(self, data_dict):
        mask_torch_data = {}

        # Convert all mask and data matrices to torch,
        # and optionally pre-load them all to GPU

        mask_matrix_args = {'dtype': torch.bool}
        data_table_args = {}

        if self.c.data_set_on_cuda:
            mask_matrix_args['device'] = self.device
            data_table_args['device'] = self.device

        # Convert mask matrices
        for mask_matrix_name in TORCH_MASK_MATRICES:
            mask_torch_data[mask_matrix_name] = torch.tensor(
                data_dict[mask_matrix_name], **mask_matrix_args)

        # Convert data table
        data_arrs = []

        for col in data_dict['data_table']:
            data_arrs.append(torch.tensor(col, **data_table_args))

        mask_torch_data['data_arrs'] = data_arrs

        # Don't need to convert to tensor -- used in batching
        mask_torch_data[
            'new_train_val_test_indices'] = data_dict[
            'new_train_val_test_indices']

        return NPTBatchDataset(
            data_dict=mask_torch_data,
            c=self.c,
            curr_cv_split=self.curr_cv_split,
            metadata=self.metadata,
            device=self.device,
            sigmas=data_dict['sigmas'])

    def cache_dataset(self, data_dict):
        dataset_path = os.path.join(
            self.cache_path, f"dataset__split={data_dict['split_idx']}.pkl")

        with open(dataset_path, 'wb') as f:
            pickle.dump(obj=data_dict, file=f)

    def update_data_attrs(self, data):
        """
        Set global dataset parameters with
            (i) metadata loaded from cache, or
            (ii) data_dict just built

        :param data: can be metadata, or data_dict
        """
        # Override number of CV splits if dataset has a fixed test set
        if data['fixed_test_set_index'] is not None:
            print(
                'Loaded metadata for fixed test set. '
                'n_cv_splits set to 1.')
            self.n_cv_splits = 1

    """Preprocessing: Data Generation Helper Functions"""

    def get_data_dict(self):
        # Get Data
        data_dict = self._dataset.get_data_dict(
            force_disable_auroc=self.c.data_set in IMAGE_DATASETS)
        assert np.intersect1d(
            data_dict['num_features'], data_dict['cat_features']).size == 0
        return data_dict

    """Preprocessing: Main Functions"""

    def tabnet_preprocess_dataset(self, data_dict):
        """Avoid caching dataset format for TabNet, because
        they are not used for other models.

        Do processing in memory.
        """
        data_dict['data_arrs'] = data_dict['data_table']
        del data_dict['data_table']
        return data_dict

    def tabnet_preprocess_metadata(self, data_dict):
        """Avoid caching metadata format for TabNet, because
        they are not used for other models.

        Do processing in memory.
        """
        metadata_dict = {
            key: data_dict[key]
            for key in METADATA_FIELDS}
        self.metadata = metadata_dict

    def run_preprocessing_and_caching(self):
        """
        Sets the self.dataset_gen attribute with a generator of custom
        TensorDataset objects.
        """
        # Cached datasets are uniquely determined by
        # 1. c.exp_seed
        # 2. cv_split index

        if not self.c.data_force_reload:
            # Try loading metadata
            try:
                self.metadata = self.load_metadata()

                # Set n_cv_split attrs
                self.update_data_attrs(data=self.metadata)

            except FileNotFoundError:
                pass

        tabnet_mode = (self.c.model_class == 'sklearn-baselines' and
                       self.c.sklearn_model == 'TabNet')
        image_data = (self.c.data_set in IMAGE_DATASETS)

        if image_data:
            data_dict = self.get_data_dict()
            self.metadata = self.get_image_metadata(data_dict)
            image_dataloaders = load_image_dataloaders(self.c)
            data_dict.update(image_dataloaders)
            return iter([
                NPTBatchDataset(
                    data_dict=data_dict,
                    c=self.c,
                    curr_cv_split=0,
                    metadata=self.metadata,
                    device=self.device,
                    sigmas=None)])

        if tabnet_mode:
            data_dicts = []

        # Skip dataset generation and caching if files are available
        if not self.are_datasets_cached():
            # Load data dict information
            data_dict = self.get_data_dict()

            # Set n_cv_split attr
            self.update_data_attrs(data=data_dict)

            # Override number of CV splits if dataset has a fixed test set
            if data_dict['fixed_test_set_index'] is not None:
                print('Fixed test set provided. n_cv_splits set to 1.')
                self.n_cv_splits = 1

            # Override number of CV splits if dataset has fully fixed indices
            if data_dict['fixed_split_indices'] is not None:
                print('Fixed train/val/test indices provided. '
                      'n_cv_splits set to 1.')
                self.n_cv_splits = 1

            self.dataset_gen = (
                self.generate_classification_regression_dataset(data_dict))

            for split_idx, data_dict in enumerate(self.dataset_gen):
                encoded_data = encode_data_dict(
                     data_dict=data_dict, c=self.c)
                if tabnet_mode:
                    (data_dict['data_table'],
                     data_dict['input_feature_dims'],
                     data_dict['standardisation'],  # Include mean and std
                     data_dict['sigmas'], data_dict['cat_dims']) = encoded_data
                else:
                    (data_dict['data_table'],
                     data_dict['input_feature_dims'],
                     data_dict['standardisation'],  # Include mean and std
                     data_dict['sigmas']) = encoded_data

                # For first index, cache metadata (generic to all CV splits)
                if split_idx == 0:
                    if tabnet_mode:
                        self.tabnet_preprocess_metadata(data_dict)
                    else:
                        self.cache_metadata(data_dict)

                if tabnet_mode:
                    data_dicts.append(
                        self.tabnet_preprocess_dataset(data_dict))
                else:
                    self.cache_dataset(data_dict)

            if not tabnet_mode:
                self.metadata = self.load_metadata()

        if tabnet_mode:
            return iter(data_dicts)
        else:
            return self.load_datasets()

    def generate_classification_regression_dataset(self, data_dict):
        """
        TODO: docstring
        """
        c = self.c
        data_table = data_dict['data_table']
        missing_matrix = data_dict['missing_matrix']
        cat_target_cols = data_dict['cat_target_cols']
        num_target_cols = data_dict['num_target_cols']
        N = data_dict['N']
        D = data_dict['D']
        cat_features = data_dict['cat_features']
        num_features = data_dict['num_features']
        auroc_setting = data_dict['auroc_setting']
        fixed_test_set_index = data_dict['fixed_test_set_index']
        fixed_split_indices = data_dict['fixed_split_indices']

        # Construct train-val-test generator

        # For a single categorical target column, use stratified KFold
        # For all other cases (e.g. many numerical, many categoricals/
        # numericals, single numerical) use standard KFold
        should_stratify = (
            len(cat_target_cols) == 1 and len(num_target_cols) == 0)

        if c.debug_no_stratify:
            should_stratify = False

        if should_stratify:
            target_col_arr = data_table[:, cat_target_cols[0]]
        else:
            target_col_arr = np.arange(N)

        if fixed_split_indices is not None:
            print('Using fixed train/val/test split indices from dataset.')
            train_val_test_splits = [fixed_split_indices]
        else:
            # If a fixed test set index is specified,
            # generates only one train-test split
            train_val_test_splits = get_class_reg_train_val_test_splits(
                target_col_arr, c,
                should_stratify=should_stratify,
                fixed_test_set_index=fixed_test_set_index)

        for split_idx, (train_val_test_indices) in enumerate(
                train_val_test_splits):
            if split_idx >= c.exp_n_runs:
                # TODO: confirm shortcircuits working
                # Number of runs is upper bounded by c.exp_n_runs.
                # Max number of runs is given by test proportion and resulting
                # number of CV folds.
                break

            # Sort data, s.t. train, val, test will be stacked after each
            # other. This will make the code much easier to deal with in the
            # production setting, and should not affect our model, since we
            # are equivariant wrt. rows.
            data_table = np.concatenate([
                data_table[train_val_test_indices[0]],
                data_table[train_val_test_indices[1]],
                data_table[train_val_test_indices[2]]], axis=0)

            lens = np.cumsum([0] + [len(i) for i in train_val_test_indices])
            new_train_val_test_indices = [
                list(range(lens[i], lens[i + 1]))
                for i in range(len(lens) - 1)]
            row_boundaries = {
                'train': lens[1], 'val': lens[2], 'test': lens[3]}

            if c.debug_leakage:
                # *** only works for breast cancer *** 
                # OPTION 1: flip val/test labels
                # labels = data_table[row_boundaries['train']:][:, 0]
                # flip_map = dict(B='M', M='B')
                # labels_flipped = [flip_map[label] for label in labels]
                # assert np.all(labels != labels_flipped)
                # data_table[row_boundaries['train']:][:, 0] = labels_flipped
                # OPTION 2: randomize instead
                # labels_flipped = np.random.choice(labels, len(labels))
                # data_table[row_boundaries['train']:][:, 0] = labels_flipped
                # OPTION 3: fill data with random stuff
                # Model should only be able to learn train by heart
                data_table[:][:, 1:] = (
                    np.random.normal(
                        size=data_table[:][:, 1:].shape)
                )

            # Build train, val, test bit matrices -- 1's where labels are
            train_mask_matrix, val_mask_matrix, test_mask_matrix = [
                get_matrix_from_rows(
                    rows=dataset_mode_rows,
                    cols=cat_target_cols + num_target_cols,  # Both lists
                    N=N, D=D)
                for dataset_mode_rows in new_train_val_test_indices]

            # Need to rebuild missing matrix with new index ordering
            new_missing_matrix = missing_matrix[
                np.concatenate(
                    [indices for indices in new_train_val_test_indices])]

            bert_mask_matrix = ~(
                    train_mask_matrix | val_mask_matrix |
                    test_mask_matrix | missing_matrix)

            # There should be no overlap between the matrices.
            # Also no entries should be missed. This assert checks for that.
            assert np.array_equal(
                train_mask_matrix ^ val_mask_matrix ^ test_mask_matrix ^
                new_missing_matrix ^ bert_mask_matrix, np.ones((N, D)))
            assert not np.array_equal(
                train_mask_matrix, np.zeros((N, D)))
            assert not np.array_equal(
                val_mask_matrix, np.zeros((N, D)))
            assert not np.array_equal(
                test_mask_matrix, np.zeros((N, D)))
            assert not np.array_equal(
                bert_mask_matrix, np.zeros((N, D)))

            data_dict = dict(
                split_idx=split_idx,
                N=N,
                D=D,
                data_table=data_table,
                cat_features=cat_features,
                num_features=num_features,
                cat_target_cols=cat_target_cols,
                num_target_cols=num_target_cols,
                auroc_setting=auroc_setting,
                missing_matrix=new_missing_matrix,
                train_mask_matrix=train_mask_matrix,
                val_mask_matrix=val_mask_matrix,
                test_mask_matrix=test_mask_matrix,
                # bert_mask_matrix=bert_mask_matrix, # We don't actually need now, it will be computed in batch dataset
                original_dataset_train_val_test_indices=(
                    train_val_test_indices),
                new_train_val_test_indices=new_train_val_test_indices,
                row_boundaries=row_boundaries,
                fixed_test_set_index=fixed_test_set_index)

            yield data_dict


class NPTDataset(torch.utils.data.Dataset):
    """
    Distributed data loading doesn't play well with IterableDatasets --
    we must explicitly materialize the batch_iter from our BatchDataset
    (see batch_dataset.py).
    """
    def __init__(self, dataset: ColumnEncodingDataset):
        super(NPTDataset).__init__()
        self.cache_path = dataset.cache_path
        self.metadata = dataset.metadata
        self.batch_iter = None

        # Necessary due to wandb config pickle issues in multiprocessing
        # self.dataset.c = None

    def __iter__(self):
        return iter(self.batch_iter)

    def __len__(self):
        return len(self.batch_iter)

    def __getitem__(self, idx):
        return self.batch_iter[idx]

    def materialize(self, cv_dataset):
        self.batch_iter = list(cv_dataset)
