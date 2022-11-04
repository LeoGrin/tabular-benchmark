import warnings
from collections import OrderedDict, defaultdict

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d

import torch
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

collate_with_pre_batching_err_msg_format = (
    "collate_with_pre_batched_map: "
    "batch must be a list with one map element; found {}")


def collate_with_pre_batching(batch):
    r"""
    Collate function used by our PyTorch dataloader (in both distributed and
    serial settings).

    We avoid adding a batch dimension, as for NPT we have pre-batched data,
    where each element of the dataset is a map.

    :arg batch: List[Dict] (not as general as the default collate fn)
    """
    if len(batch) > 1:
        raise NotImplementedError

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, container_abcs.Mapping):
        return elem  # Just return the dict, as there will only be one in NPT

    raise TypeError(collate_with_pre_batching_err_msg_format.format(elem_type))


# TODO: batching over features?

class StratifiedIndexSampler:
    def __init__(
            self, y, n_splits, shuffle=True, label_col=None,
            train_indices=None):
        self.y = y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.label_col = label_col
        self.train_indices = train_indices
        if label_col is not None and train_indices is not None:
            self.stratify_class_labels = True
            print('Stratifying train rows in each batch on the class label.')
        else:
            self.stratify_class_labels = False

    def _make_test_folds(self, labels):
        """
        Slight alterations from sklearn (StratifiedKFold)
        """
        y, n_splits, shuffle = labels, self.n_splits, self.shuffle
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (n_splits))
        if n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is less than n_splits=%d."
                           % (min_groups, n_splits)), UserWarning)

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [np.bincount(y_order[i::n_splits], minlength=n_classes)
             for i in range(n_splits)])

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(n_splits).repeat(allocation[:, k])
            if shuffle:
                np.random.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def get_stratified_test_array(self, X):
        """
        Based on sklearn function StratifiedKFold._iter_test_masks.
        """
        if self.stratify_class_labels:
            return self.get_train_label_stratified_test_array(X)

        test_folds = self._make_test_folds(self.y)

        # Inefficient for huge arrays, particularly when we need to materialize
        # the index order.
        # for i in range(n_splits):
        #     yield test_folds == i

        batch_index_to_row_indices = OrderedDict()
        batch_index_to_row_index_count = defaultdict(int)
        for row_index, batch_index in enumerate(test_folds):
            if batch_index not in batch_index_to_row_indices.keys():
                batch_index_to_row_indices[batch_index] = [row_index]
            else:
                batch_index_to_row_indices[batch_index].append(row_index)

            batch_index_to_row_index_count[batch_index] += 1

        # Keep track of the batch sizes for each batch -- this can vary
        # towards the end of the epoch, and will not be precisely what the
        # user specified. Doesn't matter because the model is equivariant
        # w.r.t. rows.
        batch_sizes = []
        for batch_index in batch_index_to_row_indices.keys():
            batch_sizes.append(batch_index_to_row_index_count[batch_index])

        return (
            X[np.concatenate(list(batch_index_to_row_indices.values()))],
            batch_sizes)

    def get_train_label_stratified_test_array(self, X):
        train_class_folds = self._make_test_folds(
            self.label_col[self.train_indices])

        # Mapping from the size of a stratified batch of training rows
        # to the index of the batch.
        train_batch_size_to_train_batch_indices = defaultdict(list)

        # Mapping from a train batch index to all of the actual train indices
        train_batch_index_to_train_row_indices = OrderedDict()

        for train_row_index, train_batch_index in enumerate(train_class_folds):
            if (train_batch_index not in
                    train_batch_index_to_train_row_indices.keys()):
                train_batch_index_to_train_row_indices[
                    train_batch_index] = [train_row_index]
            else:
                train_batch_index_to_train_row_indices[
                    train_batch_index].append(train_row_index)

        for train_batch_index, train_row_indices in (
                train_batch_index_to_train_row_indices.items()):
            train_batch_size_to_train_batch_indices[
                len(train_row_indices)].append(train_batch_index)

        test_folds = self._make_test_folds(self.y)

        # Mapping our actual batch indices to the val and test rows which
        # have been successfully assigned
        batch_index_to_val_test_row_indices = OrderedDict()

        # Mapping our actual batch indices to the total number of row indices
        # in each batch. We will have to assign the stratified train batches
        # to fulfill this constraint.
        batch_index_to_row_index_count = defaultdict(int)

        # Mapping our actual batch indices to how many train spots are
        # "vacant" in each batch. These we will fill with our stratified
        # train batches.
        batch_index_to_train_row_index_count = defaultdict(int)

        for row_index, (batch_index, dataset_mode) in enumerate(
                zip(test_folds, self.y)):
            batch_index_to_row_index_count[batch_index] += 1

            if dataset_mode == 0:  # Train
                batch_index_to_train_row_index_count[batch_index] += 1
            else:
                if batch_index not in (
                        batch_index_to_val_test_row_indices.keys()):
                    batch_index_to_val_test_row_indices[
                        batch_index] = [row_index]
                else:
                    batch_index_to_val_test_row_indices[
                        batch_index].append(row_index)

        # For all of our actual batches, let's find a suitable batch
        # of stratified training data for us to use.
        for batch_index, train_row_index_count in batch_index_to_train_row_index_count.items():
            try:
                train_batch_index = (
                    train_batch_size_to_train_batch_indices[
                        train_row_index_count].pop())
            except Exception as e:
                raise e
            batch_index_to_val_test_row_indices[batch_index] += (
                train_batch_index_to_train_row_indices[train_batch_index])

        for train_batch_arr in train_batch_size_to_train_batch_indices.values():
            if len(train_batch_arr) != 0:
                raise Exception

        batch_sizes = []
        for batch_index in batch_index_to_val_test_row_indices.keys():
            batch_sizes.append(batch_index_to_row_index_count[batch_index])

        batch_order_sorted_row_indices = X[
            np.concatenate(list(batch_index_to_val_test_row_indices.values()))]
        assert (
            len(set(batch_order_sorted_row_indices)) ==
            len(batch_order_sorted_row_indices))
        return batch_order_sorted_row_indices, batch_sizes
