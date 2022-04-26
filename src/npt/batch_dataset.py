import numpy as np
import torch
import torch.nn.functional as F

from npt.mask import mask_data_for_dataset_mode
from npt.utils.batch_utils import StratifiedIndexSampler
from npt.utils.cv_utils import DATASET_ENUM_TO_MODE
from npt.utils.encode_utils import torch_cast_to_dtype

"""
Key format:
(is_semi_supervised, dataset_mode)
is_semi_supervised: bool
dataset_mode: str, in ['train', 'val', 'test']

Value format:
Tuple[batch_modes, mode_balance]
    batch_modes: List[int], each int corresponds to
        train, val, test being present, respectively. 
        (e.g. [0, 1] = train, val)
    mode_balance: bool, true if we should mode balance the batches with
        CV stratification.
"""
BATCHING_SETTINGS_MAP = {
    # SSL Class/Reg: makes sense to stratify train vs val vs test if poss #
    (True, 'train'): ([0, 1, 2], True),
    (True, 'val'): ([0, 1, 2], True),
    (True, 'test'): ([0, 1, 2], True),

    # Prod Class/Reg: makes sense to stratify train vs val vs test if poss #
    (False, 'train'): ([0], False),
    (False, 'val'): ([0, 1], True),
    (False, 'test'): ([0, 1, 2], True)
}


class NPTBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
            self, data_dict, c, curr_cv_split, metadata, device, sigmas):
        self.c = c
        self.curr_cv_split = curr_cv_split
        self.metadata = metadata
        self.target_cols = list(
            sorted(
                self.metadata['cat_target_cols'] +
                self.metadata['num_target_cols']))
        self.sigmas = sigmas
        self.valid_modes = ['train', 'val', 'test']
        self.device = device

        if self.c.exp_batch_size == -1:
            self.batch_size = self.metadata['N']
            self.batching_enabled = False
        else:
            self.batch_size = self.c.exp_batch_size
            self.batching_enabled = True

        if self.c.data_set in ['cifar10']:
            self.is_image_dataset = True
            self.prepare_image_dataset(data_dict)
        else:
            self.is_image_dataset = False

        # The underlying data matrices of the dataset, which are stored
        # entirely on the CUDA device if c.data_set_on_cuda = True
        # These are never altered (we purely copy these, as opposed to
        # making views which might be altered downstream).
        self.data_dict = data_dict

        # Use model setting information to filter indices
        self.dataset_mode_to_batch_settings = {
            dataset_mode: self.get_batch_indices(dataset_mode=dataset_mode)
            for dataset_mode in self.valid_modes}

        self.dataset_mode_to_dataset_len = {
            dataset_mode: self.dataset_mode_to_batch_settings[dataset_mode][0]
            for dataset_mode in self.valid_modes}

        # Number of batches is dependent on dataset mode
        self.mode_to_n_batches = {
            dataset_mode: self.get_mode_n_batches(dataset_mode=dataset_mode)
            for dataset_mode in self.valid_modes}

        # Based on is_semi_supervised
        #   we construct our mask matrices for each of the dataset_modes.
        # These are made once and then fixed for the cross-validation split.
        # They are stored entirely on the CUDA device
        #   if c.data_set_on_cuda = True.
        # self.mode_masks takes the form:
        # self.mode_masks = {
        #   'train': (train_mode_mask_matrix, train_bert_mask_matrix),
        #   'val': (val_mode_mask_matrix, val_bert_mask_matrix),
        #   'test': (test_mode_mask_matrix, test_bert_mask_matrix)}
        # {dataset_mode}_mode_mask_matrix: where the target labels are, and we
        #   must mask. NOTE that this might not be precisely where we wish
        #   to compute the loss (e.g. in SSL classification at train time,
        #   we only wish to compute the loss over train rows).
        #   That is captured in the target_loss_matrix.
        # {dataset_mode}_bert_mask_matrix:
        #   ~(missing_matrix | {dataset_mode}_mode_mask_matrix)
        #   where we can apply the MLM objective using BERT-style masking.

        self.mode_masks = self.construct_mode_matrices()

        # Attributes set for each train, val, and test within an epoch/mode

        self.epoch = None
        self.dataset_mode = None
        self.row_index = 0  # Used to iterate through rows
        self.batch_index = 0  # Used to retrieve batch sizes
        self.n_batches = None
        self.dataset_len = None
        self.mode_mask_matrix = None
        self.target_loss_matrix = None
        self.bert_mask_matrix = None
        self.label_mask_matrix = None
        self.augmentation_mask_matrix = None
        self.masked_tensors = None
        self.data_arrs = None

        # Handles edge cases of last batches being diff size
        # when we use the StratifiedSampler
        self.batch_sizes = None

    def prepare_image_dataset(self, data_dict):
        if self.c.data_set not in ['cifar10']:
            raise NotImplementedError

        if self.c.data_set == 'cifar10':
            self.num_classes = 10

        print(f'Detected {self.c.data_set} dataset. '
              f'Setting num_classes = {self.num_classes}.')

        n_train_rows = 45000
        n_val_rows = 5000
        n_test_rows = 10000
        lens = [0, n_train_rows, n_val_rows, n_test_rows]
        lens = np.cumsum(lens)
        data_dict['new_train_val_test_indices'] = [
            list(range(lens[i], lens[i + 1]))
            for i in range(len(lens) - 1)]
        train_mask_matrix = torch.zeros(
            (self.metadata['N'], self.metadata['D']), dtype=torch.bool)
        val_mask_matrix = torch.zeros(
            (self.metadata['N'], self.metadata['D']), dtype=torch.bool)
        test_mask_matrix = torch.zeros(
            (self.metadata['N'], self.metadata['D']), dtype=torch.bool)

        # Assume targets are in final column
        train_mask_matrix[:n_train_rows, -1] = True
        val_mask_matrix[n_train_rows:n_train_rows+n_val_rows, -1] = True
        test_mask_matrix[-n_test_rows:, -1] = True

        print('Constructed train, val, test binary matrices with n_targets:')
        print(train_mask_matrix.sum())
        print(val_mask_matrix.sum())
        print(test_mask_matrix.sum())
        data_dict['train_mask_matrix'] = train_mask_matrix
        data_dict['val_mask_matrix'] = val_mask_matrix
        data_dict['test_mask_matrix'] = test_mask_matrix

        self.trainloader = data_dict['trainloader']
        self.validloader = data_dict['validloader']
        self.testloader = data_dict['testloader']
        return data_dict

    def load_image_dataset(self, epoch):
        # Data will already be augmented and encoded
        # Need to add zero mask tokens
        print(f'Loading image dataset at epoch {epoch}.')
        loaders = [self.trainloader, self.validloader, self.testloader]
        data = []
        labels = []
        rows_count = [0, 0, 0]

        for dataset_mode, loader in enumerate(loaders):
            for image_batch, label_batch in loader:
                n_examples = image_batch.shape[0]
                image_batch = image_batch.contiguous().view(
                    n_examples, -1)
                data.append(image_batch)
                labels.append(label_batch)
                rows_count[dataset_mode] += n_examples

        data = torch.cat(data)
        labels = torch.cat(labels)
        logging_str = 'Loaded image dataset with '
        for dataset_mode, row_count in zip(['train', 'val', 'test'], rows_count):
            logging_str += f'{row_count} {dataset_mode} rows | '

        print(logging_str)

        # Iterate through individual pixels, adding mask dimension
        data_arrs = []
        for col_index in range(data.shape[1]):
            col = data[:, col_index]
            zero_col = torch.zeros(col.shape[0])
            col = torch.stack((col, zero_col), -1)
            data_arrs.append(col)

        # One hot encode the labels, add the mask dimension
        labels = F.one_hot(labels, self.num_classes)
        label_zero_col = torch.zeros((labels.shape[0], 1))
        label_col = torch.cat((labels, label_zero_col), dim=-1)
        data_arrs.append(label_col)
        print(f'Finished loading.')
        self.data_dict['data_arrs'] = data_arrs

    def get_mode_n_batches(self, dataset_mode):
        n_batches = int(np.ceil(
            self.dataset_mode_to_dataset_len[dataset_mode] /
            self.c.exp_batch_size))
        return n_batches

    def compute_train_val_test_offsets(self):
        lens = np.cumsum(
            [0] +
            [len(i) for i in self.data_dict['new_train_val_test_indices']])
        return [lens[1], lens[2], lens[3]]

    def set_mode(self, mode, epoch):
        assert mode in self.valid_modes
        self.dataset_mode = mode
        self.epoch = epoch
        self.n_batches = self.mode_to_n_batches[mode]
        self.dataset_len = self.dataset_mode_to_dataset_len[mode]

        if self.c.verbose:
            print(
                f'Loading {mode} batches for CV split '
                f'{self.curr_cv_split + 1}, epoch {self.epoch + 1}.')

        if self.is_image_dataset and (
            mode == 'train' or (
                mode == 'val' and self.c.debug_eval_row_interactions)):
            self.load_image_dataset(epoch)

        self.batch_gen()

        if self.c.verbose:
            print('Successfully loaded batch.')

    def get_batch_indices(self, dataset_mode):
        """
        Batch indices are determined by the dataset_mode and whether or not
        we are doing SSL.
        :return: Tuple[n_rows, batch_modes, mode_indices, stratified_sampler]
            n_rows: int, the number of rows used in this mode setting. e.g. for
                SSL, will be all of the rows available.
            batch_modes: List[int], dictates if train/val/test rows are used
                in this mode. e.g. [0, 1, 2] indicates that train, val, and
                test rows are used in this mode.
            mode_row_indices: np.array(dtype=np.int)
                All row indices from the base data that must be shuffled in
                this mode.
            stratified_sampler: if we should mode balance the batches, this
                class will perform the stratified batch sampling.
                Done in all cases if the user has specified
                c.exp_batch_mode_balancing.
        """
        batch_modes, mode_balance = BATCHING_SETTINGS_MAP[(
            self.c.model_is_semi_supervised,
            dataset_mode
        )]
        mode_balance = mode_balance and self.c.exp_batch_mode_balancing

        # We will add all row indices used in this data mode to this array,
        # and concatenate.
        mode_indices = []

        # We can simply add the train_val_test_indices that are
        # specified in the batch_modes (this tells us the batching logic).
        for mode_index, arr in enumerate(
                self.data_dict['new_train_val_test_indices']):
            if mode_index in batch_modes:
                mode_indices.append(arr)


        # At this point, mode_indices has up to 3 np.arrays (for each of
        # train, val, and test rows).
        # If that is the case, and we are doing stratified batching on the
        # mode (denoted by mode_balance), we need to construct indicators
        # of the modes as follows.
        n_rows = 0
        mode_indicators = []
        stratified_sampler = None

        if len(mode_indices) > 1 and mode_balance and self.batching_enabled:
            # mode_indicator is 0 for train, 1 for val, 2 for test
            # arr_index is used to lookup the appropriate np.array in
            # the mode_indices array.
            for arr_index, mode_indicator in enumerate(batch_modes):
                mode_index_arr = mode_indices[arr_index]
                n_rows_in_mode = len(mode_index_arr)
                n_rows += n_rows_in_mode
                mode_indicators.append([mode_indicator] * n_rows_in_mode)

            # Concatenate the mode indicators
            mode_indicators = np.concatenate(mode_indicators)

            # Perform stratified batching
            bs = self.c.exp_batch_size
            n_splits = int(np.ceil(n_rows / bs))

            extra_args = {}

            # Constraints:
            # We want to class balance within batches
            # We can -- i.e., the dataset is single-target classification
            # We are at `dataset_mode` time. Are we even going to have
            # train rows to class balance?
            if (self.c.exp_batch_class_balancing and
                    self.can_class_balance() and 0 in batch_modes):
                extra_args['label_col'] = self.get_label_column()
                extra_args['train_indices'] = mode_indices[0]

            stratified_sampler = StratifiedIndexSampler(
                y=mode_indicators, n_splits=n_splits, shuffle=True,
                **extra_args)

        # Concatenate together mode_indices (which index into our matrices)
        mode_indices = np.concatenate(mode_indices)
        n_rows = len(mode_indices)

        return n_rows, batch_modes, mode_indices, stratified_sampler

    def construct_mode_matrices(self):
        """
        Our mode_mask_matrices determine where the labels ought to be
        masked out in our model inputs. They are of critical importance
        to avoid e.g. providing test labels to the model at train time
        during semi-supervised learning.

        The mode_bert_mask_matrices are computed as:
             ~(mode_mask_matrix | missing_matrix)
         and tell us where we can apply BERT-style masking.

        We use the batch_modes array for each training mode of train, val,
        and test, which tell us altogether which labels should be masked.

        For example, in semi-supervised classification, our
            self.train_mask_matrix = (
                self.data_dict['train_mask_matrix'] |
                self.data_dict['val_mask_matrix'] |
                self.data_dict['test_mask_matrix'])
        because we must mask out labels from train, val, and test.
        """
        mode_masks = {}
        missing_matrix = self.data_dict['missing_matrix']

        for dataset_mode in self.valid_modes:
            batch_modes = self.dataset_mode_to_batch_settings[dataset_mode][1]

            # We construct the mask matrix for this mode by iterating through
            # the batch modes.
            starting_mode = DATASET_ENUM_TO_MODE[batch_modes[0]]
            mode_mask_matrix = self.data_dict[f'{starting_mode}_mask_matrix']

            if len(batch_modes) > 1:
                for batch_mode in batch_modes[1:]:
                    next_mode = DATASET_ENUM_TO_MODE[batch_mode]
                    mode_mask_matrix = (mode_mask_matrix | self.data_dict[
                        f'{next_mode}_mask_matrix'])

            # Determine the bert_mask_matrix.
            bert_mask_matrix = ~(mode_mask_matrix | missing_matrix)
            mode_masks[dataset_mode] = (mode_mask_matrix, bert_mask_matrix)

        return mode_masks

    def get_label_column(self):
        target_col = self.data_dict[
            'data_arrs'][self.metadata['cat_target_cols'][0]]
        target_col = target_col[:, :-1]

        # Get integer corresp to each label
        labels = torch.argmax(
            torch_cast_to_dtype(target_col, 'float32'), dim=1).numpy()
        return labels

    def can_class_balance(self):
        class_balance = True

        if len(self.metadata['num_target_cols']) != 0:
            class_balance = False
        if len(self.metadata['cat_target_cols']) != 1:
            class_balance = False

        if class_balance:
            print('Class balancing minibatches (single-target dataset).')

        return class_balance

    # @profile
    def batch_gen(self):
        # Assures that all batchings across CV splits and epochs
        # have different seeds.
        _, batch_modes, row_index_order, stratified_sampler = (
            self.dataset_mode_to_batch_settings[self.dataset_mode])

        # Avoid stratifying when we have a super small batch size
        # TODO Should ideally check against number of classes (in classification)
        if stratified_sampler and self.batch_size > 10:
            row_index_order, batch_sizes = (
                stratified_sampler.get_stratified_test_array(row_index_order))
            self.batch_sizes = batch_sizes
        else:
            np.random.shuffle(row_index_order)
            self.batch_sizes = None

        # Construct tensor copies with the specified row index order
        mode_mask_matrix, mode_bert_mask_matrix = self.mode_masks[
            self.dataset_mode]

        # Specifies all the labels we must mask out in this mode
        self.mode_mask_matrix = mode_mask_matrix[row_index_order, :]

        # Specifies all the places at which we may use BERT masking, and
        # compute an augmentation loss
        self.bert_mask_matrix = mode_bert_mask_matrix[row_index_order, :]

        # Where we actually end up computing the loss in train/val/test
        mode_mask_matrix_str = f'{self.dataset_mode}_mask_matrix'
        self.target_loss_matrix = self.data_dict[mode_mask_matrix_str][
            row_index_order, :]

        # Stochastic label masking
        dataset_mode_mask_matrices = None
        if self.c.model_label_bert_mask_prob[self.dataset_mode] < 1:
            dataset_mode_mask_matrices = dict()
            for dataset_mode in ['train', 'val', 'test']:
                mode_mask_matrix_str = f'{dataset_mode}_mask_matrix'
                dataset_mode_mask_matrices[dataset_mode] = self.data_dict[
                    mode_mask_matrix_str][row_index_order, :]

        self.data_arrs = [
            col[row_index_order, :]
            for col in self.data_dict['data_arrs']]

        (self.masked_tensors, self.label_mask_matrix,
            self.augmentation_mask_matrix) = (
                mask_data_for_dataset_mode(
                    self.mode_mask_matrix,
                    dataset_mode_mask_matrices,
                    self.c,
                    self.metadata['cat_features'],
                    self.bert_mask_matrix, self.data_arrs, self.dataset_mode,
                    self.device))

        self.row_index = 0
        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.row_index >= self.dataset_len:
            raise StopIteration
        mode_mask_matrix_str = f'{self.dataset_mode}_mask_matrix'

        # If we have used a StratifiedSampler, we will have specific batch
        # sizes to use.
        if self.batch_sizes is not None:
            batch_size = self.batch_sizes[self.batch_index]
        else:
            batch_size = self.batch_size

        batch_dict = {
            # This matrix is True at all labels strictly associated with
            # the mode (e.g. train, val, test) -- we will compute a loss
            # at these entries.
            mode_mask_matrix_str: self.target_loss_matrix[
                self.row_index:self.row_index + batch_size],
            'cat_features': self.metadata['cat_features'],
            'num_features': self.metadata['num_features'],
            'data_arrs': [
                col[self.row_index:self.row_index + batch_size]
                for col in self.data_arrs],
            'masked_tensors': [
                col[self.row_index:self.row_index + batch_size]
                for col in self.masked_tensors],
            'target_cols': self.target_cols,
            'sigmas': self.sigmas
        }

        loss_indices = ['label', 'augmentation']

        for loss_index in loss_indices:
            matrix_str = f'{loss_index}_mask_matrix'
            matrix_attribute = getattr(self, matrix_str)
            if matrix_attribute is None:
                batch_dict[matrix_str] = None
            else:
                batch_dict[matrix_str] = matrix_attribute[
                    self.row_index:self.row_index + batch_size]

        self.row_index += batch_size
        self.batch_index += 1
        return batch_dict

    def __len__(self):
        return self.n_batches
