"""Augmentation and label masking."""

import numpy as np
import torch

from npt.constants import (
    DATA_MODE_TO_LABEL_BERT_MODE, DATA_MODE_TO_LABEL_BERT_FIXED)
from npt.utils.encode_utils import get_torch_dtype, get_torch_tensor_type


def apply_mask(
        data_arrs, mask_candidates, cat_features, mask_prob, c):
    """Apply mask_candidates to input data_arrs.

    Input:
        data_arrs: List of len D with entries torch. Tensor of shape NxH_j
            I.e. list of features D and for each feature j we have N rows and
            one-hot encoding dimension H_j.
        mask_candidates: Boolean torch.Tensor matrix of shape (N, D). Is True
            for entries which qualify for masking.
        cat_features: List of all categorical columns. Categorical columns are
            treated differently for masking.
        mask_prob: If mask_prob == 1, we mask all entries for which
            mask_candidates is True. If mask_prob < 1, we draw Bernoulli
            random variables at each entry. If those turn up true, we (a) set
            the masking token and (b) either zero-out or randomise the value
            of the entry. We zero-out with p=c.model_bert_mask_percentage
            and randomise with p=1-c.model_bert_mask_percentage.

    Returns:
        data_arrs: Now with masked entries.
        mask: Boolean torch.Tensor of shape (N, D). True for all masks that
            were set.

    """
    if c.data_set_on_cuda:
        device = c.exp_device
    else:
        device = 'cpu'

    num_examples = data_arrs[0].shape[0]

    # Relevant for production setting only
    if not c.model_is_semi_supervised:
        # Filter out all mask_candidates which are not present in data_arrs.
        # (I.e. at train mode, we are not given anything in test or val.)
        # Since we sort the data_arrs row-wise by train, val, test, we can be
        # sure that all mask_candidates with row indices larger than
        # need not be considered.
        mask_candidates = mask_candidates[:num_examples]

    # Mask features
    feature_mask_indices_arrs = []  # For each feature, list of masked rows

    # Shortcut for deterministic masking
    if mask_prob == 1:
        # no sampling: can set mask to mask candidates directly
        # 'mask' contains entries for which indices will be zero-ed out
        mask = mask_candidates
        # no resampling. set masks for which entries are randomized to None
        bert_random_mask = None

    # Stochastic Bert Style Masking
    else:
        # We need to extract a list of all matrix entries with mask candidates.
        # mask_entries has shape (Nm, 2).
        mask_entries = torch.nonzero(mask_candidates, as_tuple=False)
        Nm = len(mask_entries)  # number of mask candidates

        # Simulate bernoulli sampling of masks:
        # Performing Nm bernoulli samples with probability mask_prob
        # gives an expected value of
        expected = mask_prob * Nm
        # and a standard deviation of
        std = np.sqrt(Nm * mask_prob * (1 - mask_prob))
        # This gives the total number of masks sampled in our approximative
        # sampling scheme as.
        num_masks_sampled = int(
            mask_prob * Nm +
            np.random.normal(0, std))

        # Make sure this is always valid
        num_masks_sampled = max(min(num_masks_sampled, Nm), 0)

        # We now take a random subset of the total number of mask indices.
        mask_indices_indices = np.random.choice(
                np.arange(0, Nm),
                size=num_masks_sampled,
                replace=False)

        # Select valid indices from sampled mask indices.
        # This is a tensor of shape (len(mask_indices_indices), 2).
        mask_indices = mask_entries[mask_indices_indices, :]

        # Now we need to split mask_indices into those for which we want to
        # zero out and those we want to randomly resample.
        # (Bert-style masking)

        # Proportion for which to zero out.
        bert_random_mask_proportion = 1 - int(
            c.model_bert_mask_percentage * len(mask_indices))

        # Since the mask_indices have already been choiced at random
        # we can just use slicing to select the indices this time.
        bert_random_mask_indices = mask_indices[:bert_random_mask_proportion]

        # Reconstruct mask matrices from list of entries.
        mask = torch.sparse.FloatTensor(
            mask_indices.T,
            torch.ones(len(mask_indices), dtype=torch.int, device=device),
            mask_candidates.size(),
            ).to_dense().type(torch.bool)

        bert_random_mask = torch.sparse.FloatTensor(
            bert_random_mask_indices.T,
            torch.ones(
                len(bert_random_mask_indices), dtype=torch.int, device=device),
            mask_candidates.size(),
            ).to_dense().type(torch.bool)

        # Mask is never 1 where mask_candidate is 0
        assert ((mask_candidates.int() - mask.int()) < 0).sum() == 0
        # bert_random_mask is never 1 where Mask is 0
        assert ((mask.int() - bert_random_mask.int()) < 0).sum() == 0

    # Iterate over all columns and set mask.
    for col, data_arr in enumerate(data_arrs):
        # Get boolean 'mask' selection mask for this column.
        mask_col = mask[:, col]

        # If there are no masks in this column, continue.
        if mask_col.sum() == 0:
            continue

        # Zero out indices corresponding to
        # bert masking and bert random assignment.
        data_arr[mask_col, :] = 0

        # Set mask token.
        # We don't want to set this token for entries that are randomized;
        # in the analogous BERT language case, the [MASK] token is not used!
        if bert_random_mask is None:
            # If there is no bert randomization,
            # all mask entries should be given a '1' mask token,
            data_arr[mask_col, -1] = 1
            # and we are done with masking for this column
            continue

        bert_random_mask_col = bert_random_mask[:, col]

        # Determine for which entries to set the mask token.
        # If bert_random_mask_col is all False, we just mask entries determined
        # by mask_col
        bert_mask_token_col = mask_col & (~bert_random_mask_col)

        # Set mask token in last position of each feature vector,
        # for appropriate entries.
        data_arr[bert_mask_token_col, -1] = 1

        # We are done with the mask_col; only BERT randomization remains.

        if bert_random_mask_col.sum() == 0:
            # We are done w/ this col if there are
            # no BERT randomization entries.
            # (bert_random_mask can be not None and there can still be no
            # random masking to do in this column due to chance)
            continue

        # Randomize entries in bert_random_mask_col
        # For categorical features:
        #   Select random columns (one-hot-values, excluding masking token)
        #   for all selected entries in row and set to one.
        if col in cat_features:
            random_cols = torch.randint(
                low=0,
                high=data_arr.shape[1]-1,  # high is exclusive for torch
                size=[bert_random_mask_col.sum()],
                requires_grad=False)

            data_arr[bert_random_mask_col, random_cols] = 1
        # For continuous features
        #   Sample new entry values for selected entries in this row
        #   from normal distribution.
        else:
            data_dtype = get_torch_dtype(dtype_name=c.data_dtype)
            data_arr[bert_random_mask_col, 0] = torch.normal(
                mean=0, std=1,
                size=[bert_random_mask_col.sum()], dtype=data_dtype,
                device=device)

    return data_arrs, mask


def mask_data_for_dataset_mode(
        deterministic_label_masks,  # e.g. at train mask train, test, val
        stochastic_label_masks,     # e.g. contains only train for train
        c, cat_features, bert_mask_matrix,
        data_arrs, dataset_mode, device):
    """
    Mask data table for the current dataset mode (e.g. at train time,
    val time, or test time).

    This method handles the high-level logic of which entries of the
    table should get masked.

    The method is split in two parts:
    1 – TARGET MASKING: We mask the target values of the table.
    2 - FEATURE MASKING: We apply Bert-style feature masking.

    Each section assembles boolean matrices of mask_candidates, i.e.
    positions for which masks may be applied, and the actual masks
    are then applied to the data in apply_mask().
    Apply_mask() can do both deterministic (mask_prob = 1) and stochastic
    (mask_prob < 1) masking.

    1 - TARGET MASKING
    * Target masking can either be deterministic (default) or stochastic
    (largely untested). In classification/regression settings, the targets
    would correspond to the label column.

        * Deterministic target masking: We mask out all
        targets the model is not supposed to see at input. Currently, we
        always mask out all given target labels, regardless of whether
        they belong to train, val, or test and regardless of the
        dataset_mode. (I.e. the model never sees target values at input).
        * Stochastic target masking: We only mask labels out
        stochastically. I.e. the model can sometimes see the train labels at
        input. We never reveal test targets to the model.

    2 – FEATURE MASKING
    * Apply Bert-style augmentation masking on the features.

    """
    # Without copy, we overwrite true data.
    # Also need to copy exactly in this way (or possibly with deepcopy).
    input_arrs = [di.clone() for di in data_arrs]
    # for arr in input_arrs:
    #     print(arr.dtype)

    # ##################################
    # ******* 1 TARGET MASKING ********
    # ##################################

    if c.model_label_bert_mask_prob[dataset_mode] == 1:

        # Deterministic masking on all targets
        # apply_mask() takes care to only mask entries which exist in the
        # current input, therefore this code can be applied both in semi-
        # supervised and in production case.

        mask_candidates = deterministic_label_masks

        masked_arrs, label_mask_matrix = (
            apply_mask(
                data_arrs=input_arrs,
                mask_candidates=mask_candidates,
                cat_features=cat_features,
                mask_prob=1,
                c=c))
        # Set this to none because label_mask_matrix is all possible values.
        label_mask_matrix = None

        # [masked_arrs[j][i] for (i,j) in mask_candidates]
        # in the breast cancer classification setting, this now holds:
        # only the first col has been changed
        # all([np.array_equal(masked_arrs[i], input_arrs[i])
        #   for i in range(1, len(input_arrs))])
        # all entries are masked that col
        # np.all(input_arrs[0][:, :2] == 0)
        # np.all(input_arrs[0][:, 2] == 1)

    else:

        # Stochastic Masking on Some Labels
        label_prob = c.model_label_bert_mask_prob[dataset_mode]

        if (label_prob == 0) and (dataset_mode == 'train'):
            # mask none of the train labels
            # but need to make sure we tell that to the loss also
            # (no longer want to compute loss here)
            # --> empty mask_indices matrix
            stochastic_mask_indices = np.zeros_like(
                stochastic_label_masks['train'])
            masked_arrs = input_arrs
        elif label_prob > 0:
            # Stochastically mask out some targets (never test targets)

            mask_categories = DATA_MODE_TO_LABEL_BERT_MODE[dataset_mode]
            mask_candidates = torch.stack(
                [stochastic_label_masks[category] for category in
                    mask_categories]
            ).sum(0, dtype=torch.bool)

            masked_arrs, stochastic_mask_indices = apply_mask(
                data_arrs=input_arrs,
                mask_candidates=mask_candidates,
                cat_features=cat_features,
                mask_prob=label_prob,
                c=c)
            # label_mask_indices not outside mask_candidates
            assert (
                (~mask_candidates)
                & (stochastic_mask_indices)).any().item() is False
            if c.model_label_bert_mask_prob[dataset_mode] == 1:
                assert (
                    mask_candidates & stochastic_mask_indices
                    ).all().item() is True

        else:
            masked_arrs = input_arrs

            # This is when we are not stochastically masking train entries
            # at val time (train+val .. at test time).
            # i.e. all train labels are revealed at val/test time
            # there is no loss computed on them, so we don't care for the
            # purposes of loss.py
            # (i.e. this branch is only reached for label_prob == 0 at
            # val/test)
            # stochastic_mask_indices = None

        # Deterministic masking on some labels (always test targets)
        mask_categories = DATA_MODE_TO_LABEL_BERT_FIXED[dataset_mode]
        mask_candidates = torch.stack(
            [stochastic_label_masks[category] for category in
             mask_categories]
        ).sum(0, dtype=torch.bool)

        masked_arrs, _ = apply_mask(
            data_arrs=masked_arrs,
            mask_candidates=mask_candidates,
            cat_features=cat_features,
            mask_prob=1,
            c=c)

        if dataset_mode == 'train':
            # at train time, we care about the loss on the train labels that 
            # were actually masked out
            label_mask_matrix = stochastic_mask_indices
        else:
            # at val/test time, we care about the loss on val/test, which are
            # never stochastically revealed.
            label_mask_matrix = None

        # label_mask_indices is used to 'and' with the
        # 'data_dict[f'{dataset_mode}_mask_matrix']'
        # so we can just put all values that have not been revealed
        # in there

        # if label_mask_matrix is not None:
        #     label_mask_indices = label_mask_indices | tmp
        # else:
        #     label_mask_indices = tmp

    # ##################################
    # ****** 2 – FEATURE MASKING *******
    # ##################################

    if c.model_augmentation_bert_mask_prob[dataset_mode] > 0:
        masked_arrs, augmentation_mask_matrix = (
            apply_mask(
                data_arrs=masked_arrs,
                mask_candidates=bert_mask_matrix,
                cat_features=cat_features,
                mask_prob=(c.model_augmentation_bert_mask_prob[dataset_mode]),
                c=c))
    else:
        augmentation_mask_matrix = None

    # masked_arrs are already torch now -- move to GPU
    data_dtype = get_torch_tensor_type(c.data_dtype)

    masked_tensors = [
        masked_arr.type(data_dtype) for masked_arr in masked_arrs]

    if c.data_set_on_cuda:
        masked_tensors = [
            masked_arr.type(data_dtype).to(device=c.exp_device)
            for masked_arr in masked_arrs]

    return (
        masked_tensors, label_mask_matrix, augmentation_mask_matrix)
