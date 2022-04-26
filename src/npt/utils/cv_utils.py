"""Cross-validation utils."""

from collections import Counter
from enum import IntEnum

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold


class DatasetMode(IntEnum):
    """Used in batching."""
    TRAIN = 0
    VAL = 1
    TEST = 2


DATASET_MODE_TO_ENUM = {
    'train': DatasetMode.TRAIN,
    'val': DatasetMode.VAL,
    'test': DatasetMode.TEST
}

DATASET_ENUM_TO_MODE = {
    DatasetMode.TRAIN: 'train',
    DatasetMode.VAL: 'val',
    DatasetMode.TEST: 'test'
}


def get_class_reg_train_val_test_splits(
        label_rows, c, should_stratify, fixed_test_set_index):
    """"Obtain train, validation, and test indices.
    num_data = len(label_rows)

    Stratify Logic:

    Perform stratified splits if the target is a single categorical column;
    else, (even if we have multiple categorical targets, for example)
    perform standard splits.

    If fixed_test_set_index is not None,
    use the index to perform the test split
    """
    if should_stratify and label_rows.dtype == np.object:
        from sklearn.preprocessing import LabelEncoder
        # Encode the label column
        label_rows = LabelEncoder().fit_transform(label_rows)
        print('Detected an object dtype label column. Encoded to ints.')

    N = len(label_rows)
    n_cv_splits = get_n_cv_splits(c)

    if fixed_test_set_index:
        all_indices = np.arange(N)
        train_test_splits = [
            (all_indices[:fixed_test_set_index],
             all_indices[fixed_test_set_index:])]
    else:
        kf_class = StratifiedKFold if should_stratify else KFold
        kf = kf_class(
            n_splits=n_cv_splits, shuffle=True, random_state=c.np_seed)
        train_test_splits = kf.split(np.arange(N), label_rows)

    for train_val_indices, test_indices in train_test_splits:
        val_indices = []
        if c.exp_val_perc > 0:
            normed_val_perc = c.exp_val_perc / (1 - c.exp_test_perc)

            if should_stratify:
                train_val_label_rows = label_rows[train_val_indices]
            else:
                train_val_label_rows = None

            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=normed_val_perc, shuffle=True,
                random_state=c.np_seed, stratify=train_val_label_rows)
        else:
            train_indices = train_val_indices

        train_perc = len(train_indices) / N
        val_perc = len(val_indices) / N
        test_perc = len(test_indices) / N
        print(
            f'Percentage of each group: Train {train_perc:.2f} '
            f'| {val_perc:.2f} | {test_perc:.2f}')

        if c.exp_show_empirical_label_dist:
            print('Empirical Label Distributions:')
            for split_name, split_indices in zip(
                    ['train', 'val', 'test'],
                    [train_indices, val_indices, test_indices]):
                num_elem = len(split_indices)
                class_counter = Counter(label_rows[split_indices])
                class_proportions = {
                    key: class_counter[key] / num_elem
                    for key in sorted(class_counter.keys())}
                print(f'{split_name}:')
                print(class_proportions)

        yield train_indices, val_indices, test_indices


def get_n_cv_splits(c):
    return int(1 / c.exp_test_perc)  # Rounds down
