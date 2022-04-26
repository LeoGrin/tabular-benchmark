from collections import defaultdict

import torch
import torch.nn as nn
from torchmetrics.functional import auroc as lightning_auroc

from npt.utils.encode_utils import torch_cast_to_dtype


class Loss:
    """Compute losses.

    Keep track of losses in batch and epoch.

    Some complexity exists because we
    * need to take care to compute loss at the correct entries and
    * keep track of batch statistics and aggregate them into epochs. (E.g.
      normalize losses by number of predictions or calculate accuracies.)

    However, the interface from the outside is simple. The user calls
    * Loss.compute() to calculate the loss for the current
        input/output-pairs.
    * Loss.finalize_batch_losses() to obtain the loss_dict for the last input.
    * Loss.finalize_epoch_losses() to obtain the loss_dict for the current
        epoch.

    The loss_dict is a nested dictionary, where the first level of keys
    ('label' and 'augmentation') separates between losses from targets and
    losses from Bert style feature masking.
    The second level splits losses into categorical and numerical losses, and
    also offers some amenities such as accuracy or AUROC metrics.

    Loss.compute_loss() does most of the heavy lifting for the correct
    computation of losses. See the docstring for how and where we compute loss.

    """
    def __init__(
            self,
            is_minibatch_sgd, metadata, model_bert_augmentation=False,
            model_augmentation_bert_mask_prob=0.0,
            exp_tradeoff=-1,
            device=None, tradeoff_annealer=None, data_set="generic",
            exp_print_every_nth_forward=50, data_dtype=torch.long):
        """

        :param c:
        :param metadata:
        :param is_minibatch_sgd
        :param device: Must be set for distributed setting.
        :param tradeoff_annealer: Provided when there is annealing specified
            between augmentation and label loss (see config: exp_tradeoff).
        :param sigmas: Standard deviation values of training set (one per col).
        """
        self.exp_tradeoff = exp_tradeoff
        self.data_dtype = data_dtype
        self.model_augmentation_bert_mask_prob = model_augmentation_bert_mask_prob
        self.data_set = data_set
        self.model_bert_augmentation = model_bert_augmentation
        self.exp_print_every_nth_forward = exp_print_every_nth_forward


        self.tradeoff_annealer = tradeoff_annealer
        self.cross_ent_loss = nn.CrossEntropyLoss(reduction='sum')
        self.cross_ent_loss_no_sum = nn.CrossEntropyLoss(reduction='none')
        self.reset()
        self.is_minibatch_sgd = is_minibatch_sgd
        #self.device = self.c.exp_device if device is None else device
        self.device = device
        self.tradeoff_annealer = tradeoff_annealer

        self.loss_modes = ['augmentation', 'label']
        self.loss_stats = [
            'num_loss', 'num_total_preds', 'num_mse_loss', 'cat_loss',
            'cat_correct_preds', 'cat_total_preds', 'total_loss']
        # save unstandardised values here
        self.extras = ['num_mse_loss_unstd', 'num_loss_unstd']
        self.loss_stats += self.extras

        self.setup_auroc(metadata)

    def setup_auroc(self, metadata):
        if metadata.get('auroc_setting', False):
            cat_target_cols = metadata['cat_target_cols']
            print('Using AUROC in loss module.')
            self.use_auroc = True
            self.auroc_col = cat_target_cols[0]
            self.softmax = nn.Softmax(dim=1)
            self.reset_auroc()
        else:
            print('Disabled AUROC in loss module.')
            self.use_auroc = False

    def reset_auroc(self):
        """Reset helper variables for epoch-wide AUROC computation."""
        self._batch_predictions = []
        self._batch_true_vals = []

    def reset(self):
        """Reset batch and epoch losses."""
        self.batch_loss = None
        self.epoch_loss = None

    def compute(self, *args, **kwargs):
        """Compute loss and update batch losses."""

        loss_dict = self.compute_loss(*args, **kwargs)
        self.batch_loss = loss_dict

    def update_losses(self, eval_model):
        """Update losses.

        In the case of minibatch SGD, this function should only be
         called after we have already backpropagated on the batch loss,
         because we detach all those tensors prior to adding them to
         the epoch loss (no need to retain the computation graph if
         we are just logging per-epoch).

        Set batch loss to current value.

        Add current batch loss to epoch loss.
        """
        # Throw away gradient information:
        #   (I) if we are evaluating the model, or
        #   (II) if we are minibatching (because we have already
        #           backpropped, and just want to store loss info for per-epoch
        #           logging).
        if eval_model or self.is_minibatch_sgd:
            self.detach_all(self.batch_loss)

        # TODO: epoch loss only relevant if eval_model or full_batch_sg
        # TODO: could skip this computation if not relevant
        # update epoch loss to incorporate current batch
        if self.epoch_loss is None:
            self.epoch_loss = self.batch_loss
        else:
            for mode in self.epoch_loss.keys():
                for key in self.epoch_loss[mode].keys():
                    self.epoch_loss[mode][key] = (
                        self.epoch_loss[mode][key] +
                        self.batch_loss[mode][key])

    # @profile
    def compute_loss(
            self, output, ground_truth_data, data_dict, label_mask_matrix,
            augmentation_mask_matrix, dataset_mode, eval_model):
        """Compute losses by iterating over columns.

        This function iterates over all columns of the input/output, calculates
        losses at entries where masks were applied at input.

        Args:
            output: Predictions of model. list, len(D), each element j is
                torch.Tensor of shape (N, H_j).
            ground_truth_data: Ground truth data. list, len(D), each element j
                is torch.Tensor of shape (N, H_j).
            data_dict: All other data information, e.g. masking matrices.
            label_mask_matrix: Boolean matrix indicating where target values
                were masked as input to the model.
            augmentation_mask_matrix: Boolean matrix indicating where feature
                values were masked as input to the model.
            dataset_mode: str in ('train', 'val', 'test')
            eval_model: bool. If we are computing loss for evaluation purposes
                (i.e. logging purposes) also compute other metrics.
                If False, this triggers some computational shortcuts.

        We aggregate the following losses:
            Categorical:
                Cross ent, acc over features
                Cross ent, acc, auroc over labels
            Numerical:
                MSE over features
                MSE over labels
            Overall loss:
                Over features
                Over labels
        """
        self.dataset_mode = dataset_mode

        # Assemble matrices over which to compute loss for label and
        # augmentation losses.
        loss_indices = dict()

        # *** Where to compute target loss? ***
        # Only compute loss where appropriate for dataset_mode
        #   * train_labels at train time
        #   * val_labels at val time
        #   * test_labels at test time
        #   --> This is achieved by computing loss only where the respective
        #       data_dict[f'{dataset_mode}_mask_matrix'] is True.

        # If we are in "production mode", i.e. the model is not provided
        # test features at train time, the data is already sliced via the
        # BatchDataset.

        # Usually deterministic label masking
        #   (At val/test time for stochastic label masking, we do not
        #   unveil val/test labels, only unveil train/(train,val).
        #   So no need to adjust the loss here.)
        if label_mask_matrix is None:
            # Select the correct mask matrix!
            loss_indices['label'] = data_dict[f'{dataset_mode}_mask_matrix']

        # Stochastic label masking
        else:
            # Do not compute loss on targets which were revealed as inputs.
            # If it were not for stochastic label masking,
            # we would not need to input the label_mask_matrix, because
            # all info is in the {dataset_mode}_mask_matrix
            # loss_indices['label'] = (
                # data_dict[f'{dataset_mode}_mask_matrix'] &
                # label_mask_matrix)

            # At train time, we only compute loss over stochastic train labels
            # at val/test we compute loss over deterministically masked val/
            # test labels.
            loss_indices['label'] = label_mask_matrix

        # *** Where to compute feature loss? ***
        # We also compute loss on the Bert style masked features

        # Technically, we ought to check to see if the augmentation_mask_matrix
        # had no samples -- but this will never happen for a reasonably large
        # number of rows/cols per batch. We are trying to avoid a CUDA sync.
        # if augmentation_mask_matrix is None or (
        #         augmentation_mask_matrix.sum() == 0):
        if self.model_augmentation_bert_mask_prob[dataset_mode] == 0:
            loss_indices['augmentation'] = None
        else:
            loss_indices['augmentation'] = augmentation_mask_matrix

        # Initialise loss_dict
        loss_dict = {
            loss_mode: {
                key: torch.zeros((1), device=self.device)
                if 'loss' in key
                else 0 for key in self.loss_stats
            }
            for loss_mode in self.loss_modes
        }

        # Compute losses per column
        for col, (out, dat) in enumerate(zip(output, ground_truth_data)):
            is_cat = col in data_dict['cat_features']
            if not is_cat and self.data_set not in ['cifar10']:
                sigma = data_dict['sigmas'][col]
            else:
                sigma = None

            if self.model_bert_augmentation:
                # Remove mask from ground truth data
                dat = dat[:, :-1]

            for mode, mode_loss_indices in loss_indices.items():
                # Compute label and augmentation losses separately

                # Short-circuit if mode is label and col is not a label col
                if mode == 'label' and col not in data_dict['target_cols']:
                    continue

                # Necessary e.g. when aug masking is disabled for val/test
                if mode_loss_indices is None:
                    continue

                # Get row indices for which we want to compute loss in this col
                col_loss_indices = mode_loss_indices[:, col]

                # Keep track of number of predicted values in this column
                # for loss normalisation purposes.
                num_preds = col_loss_indices.sum()

                # Compute loss for selected row entries in col
                loss, extra_out = self.compute_column_loss(
                    col=col, is_cat=is_cat, output=out, data=dat,
                    eval_model=eval_model, col_mask=col_loss_indices,
                    num_preds=num_preds, sigma=sigma)

                if loss is None:
                    continue

                # Log loss and accuracy metrics
                if is_cat:
                    loss_dict[mode]['cat_loss'] += loss
                    loss_dict[mode]['cat_total_preds'] += num_preds
                    if eval_model:
                        loss_dict[mode]['cat_correct_preds'] += (
                            extra_out['cat_correct_preds'])
                else:
                    loss_dict[mode]['num_loss'] += loss
                    loss_dict[mode]['num_total_preds'] += num_preds
                    loss_dict[mode]['num_mse_loss'] += (
                        extra_out['num_mse_loss'])

                    # loss unstandardisation for regression cols
                    for extra in self.extras:
                        if extra_loss := extra_out.get(extra, False):
                            loss_dict[mode][extra] += extra_loss

        return loss_dict

    def finalize_batch_losses(self):
        """Normalise batch losses by number of predictions."""
        return self.finalize_losses(self.batch_loss, False)

    def finalize_epoch_losses(self, eval_model):
        """Normalise epoch losses and reset stored values."""
        std_dict = self.finalize_losses(self.epoch_loss, eval_model)

        if eval_model and self.use_auroc:
            # also compute AUROC metric after each batch
            auroc = self.compute_auroc()
            std_dict['label']['auroc'] = auroc
            self.reset_auroc()

        return std_dict

    def get_intermediate_epoch_losses(self):
        """
        For row corruption and other very slow tasks, we want to log
        intermediate losses after every few minibatches.
        Only difference from the above is that we don't reset stored values.
        """
        std_dict = self.finalize_losses(self.epoch_loss, eval_model=True)
        if self.use_auroc:
            auroc = self.compute_auroc()
            std_dict['abel']['auroc'] = auroc

        return std_dict

    def finalize_losses(self, raw_dict, eval_model):
        """Before we backpropagate or log, we need to finalise the losses.

        * calculate total loss by weighing label and augmentation losses and
            normalising by the total number of predictions made.
        * if we are evaluating model, also compute losses and accuracies for
            the 'label' and 'augmentation' categories separately

        We can only do this directly before backprop or logging, since only
        then do we know the total number of predictions, for example because
        we aggregate losses accumulated over several minibatches.

        """
        std_dict = defaultdict(dict)

        if eval_model:
            self.detach_all(raw_dict)

        # *** Total Loss ***
        # Compute total loss. This is used for backpropagation
        # Trade-off loss on target columns and loss from augmentation masking.
        std_dict['total_loss'] = self.balance_self_supervision(raw_dict)

        if not eval_model and not self.exp_print_every_nth_forward:
            return std_dict

        # *** Logging Extra Losses ***
        # Has no bearing on backprop, just for logging purposes.
        for mode in raw_dict.keys():
            # keys are subset of {augmentation, label}
            # Normalize losses in the different modes and calculate accuracies.
            cat_preds = float(raw_dict[mode]['cat_total_preds'])
            num_preds = float(raw_dict[mode]['num_total_preds'])
            total_preds = cat_preds + num_preds

            if total_preds > 0:
                for add in ['', '_unstd']:
                    std_dict[mode][f'total_loss{add}'] = (
                        (raw_dict[mode]['cat_loss']
                         + raw_dict[mode][f'num_loss{add}'])
                        / total_preds)
            if num_preds > 0:
                for loss in ['num_loss', 'num_mse_loss'] + self.extras:
                    std_dict[mode][loss] = raw_dict[mode][loss] / num_preds
            if cat_preds > 0:
                out_names = ['cat_loss', 'cat_accuracy']
                in_names = ['cat_loss', 'cat_correct_preds']
                for out_loss, in_loss in zip(out_names, in_names):
                    std_dict[mode][out_loss] = (
                        raw_dict[mode][in_loss] / cat_preds)

        return std_dict

    def balance_self_supervision(self, raw_dict):
        """Balance weights from augmentation loss and label loss.

        If tradeoff is specified as -1, we normalise augmentation and label
        predictions jointly (i.e. sum their losses, and divide by total number
        of predictions.

        Otherwise, we normalise augmentation and label predictions separately,
        and then combine these two loss categories with a convex combination.
        """
        tradeoff_setting = self.exp_tradeoff

        # Total number of predictions across all modes and data types
        if tradeoff_setting == -1:
            normalisation = (
                raw_dict['label']['cat_total_preds']
                + raw_dict['label']['num_total_preds']
                + raw_dict['augmentation']['cat_total_preds']
                + raw_dict['augmentation']['num_total_preds'])

            # Total loss over all modes and data types.
            total_loss = (
                raw_dict['label']['cat_loss']
                + raw_dict['label']['num_loss']
                + raw_dict['augmentation']['cat_loss']
                + raw_dict['augmentation']['num_loss'])
            return total_loss / normalisation
        else:
            # There may be no losses to balance if we don't do augmentation

            # We could check this explicitly with the below, but this forces
            # a CUDA sync point
            # if aug_loss.isnan():
            #     aug_loss = 0
            # if label_loss.isnan():
            #     label_loss = 0

            # Instead, we check as follows:
            label_loss = (
                (raw_dict['label']['cat_loss'] +
                 raw_dict['label']['num_loss'])
                /
                (raw_dict['label']['cat_total_preds'] +
                 raw_dict['label']['num_total_preds']))
            if self.model_augmentation_bert_mask_prob[
                    self.dataset_mode] == 0:
                aug_loss = 0
            else:
                aug_loss = (
                    (raw_dict['augmentation']['cat_loss'] +
                     raw_dict['augmentation']['num_loss'])
                    /
                    (raw_dict['augmentation']['cat_total_preds'] +
                     raw_dict['augmentation']['num_total_preds']))

            tradeoff = self.tradeoff_annealer.curr_tradeoff
            total_loss = tradeoff * aug_loss + (1 - tradeoff) * label_loss
            return total_loss

    def compute_column_loss(
            self, col, is_cat, output, data, eval_model, col_mask, num_preds,
            sigma=None):
        """Compute loss for selected rows in a single column.

        Args:
            col (int): Index of current column.
            is_cat (bool): Is column categorical? If not, is continuous.
            output (torch.Tensor): Predictions from model in that column.
            data (torch.Tensor): True data for that column.
            eval_model (bool): Aggregate more data for logging purposes.
            col_mask (torch.Tensor): entries for which we did use a mask,
                and therefore should compute a loss.
            num_preds (torch.Tensor): Number of entries for which we are
                making a prediction.
            mode (str): Are we computing 'label' or 'augmentation' loss.
            sigma (float): Standard deviation of training column for this col.
                This is needed to provide 'unstandardised' MSE values.

        Returns:
            loss (torch.Tensor): Loss value for that column.
            cat_correct_preds (torch.Tensor): Number of correct predictions for
                accuracy calculation. None for regression.
        """
        extra_out = dict()

        if is_cat:
            # Cross-entropy loss does not expect one-hot encoding.
            # Instead, convert data to integer list of correct labels:
            # long_data: list of ints, each entry in [0, ..., C-1]
            long_data = torch.argmax(
                torch_cast_to_dtype(obj=data, dtype_name=self.data_dtype),
                dim=1).to(device=self.device)

            # Compute sum of cross_entropy losses.
            loss = self.cross_ent_loss_no_sum(output, long_data)

            # Only count the loss for entries that were masked
            loss = loss * col_mask

            # We use the unreduced loss above - reduce here
            loss = loss.sum()

            # We do this infrequently, because it forces a CUDA sync
            if eval_model:
                if col_mask.sum() == 0:
                    return None, None

                valid_long_data = long_data[col_mask]
                valid_long_output = output[col_mask]

                # Record number of correct predictions
                valid_long_output = torch.argmax(
                    valid_long_output, dim=1).to(device=self.device)
                cat_correct_preds = (
                    valid_long_output == valid_long_data).sum()
                extra_out.update(cat_correct_preds=cat_correct_preds)

            # aggregate predictions over batch for auroc computation
            if eval_model and self.use_auroc and (col == self.auroc_col):
                self._batch_predictions.append(output[col_mask].detach())
                self._batch_true_vals.append(long_data[col_mask].detach())

        else:
            # Apply the invalid entries multiplicatively, so we only
            # tabulate an MSE for the entries which were masked
            output = col_mask * output.squeeze()
            data = col_mask * data.squeeze()

            loss = torch.sum(torch.square((output - data)))
            extra_out['num_mse_loss'] = loss.detach()

            if eval_model and self.data_set not in ['cifar10']:
                # also record unnormalised MSE values at evaluation time
                # on regression columns
                mse_unstd = extra_out['num_mse_loss'] * sigma**2
                extra_out[self.extras[0]] = mse_unstd
                extra_out[self.extras[1]] = mse_unstd

        return loss, extra_out

    def compute_auroc(self):
        """Compute auroc loss metric for predictions aggregated over batch."""
        preds = torch.cat(self._batch_predictions, dim=0)
        preds = self.softmax(preds)
        true = torch.cat(self._batch_true_vals, dim=0)
        if torch.sum(true) == 0:
            # No positive samples in targets
            # true positive value should be meaningless
            auroc = 0
        else:
            try:
                auroc = lightning_auroc(preds[:, 1], true)
            except ValueError as e:
                print('AUROC computation failure.')
                print(e)
                auroc = 0

        return auroc

    def detach_all(self, raw_dict):
        # outer dict
        for mode in raw_dict.keys():
            for key, value in raw_dict[mode].items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    raw_dict[mode][key] = value.detach()
