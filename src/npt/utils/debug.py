import torch
import numpy as np


def modify_data(c, batch_dict, dataset_mode, num_steps):
    """Modify data for debugging row interactions in synthetic experiments."""

    if 'protein-duplicate' in c.debug_row_interactions_mode:
        return protein_duplicate(
            c, batch_dict, dataset_mode, c.debug_row_interactions_mode)
    else:
        raise ValueError


def corrupt_rows(c, batch_dict, dataset_mode, row_index):
    """Corrupt rows:
    (i) Duplication experiments -- find the duplicated row of the specified
        `row_index`. Flip its label.
    (ii) Standard datasets -- for each column, apply an independent permutation
        over entries in all rows other than row `row_index`.
    """
    if (c.debug_row_interactions and
            c.debug_row_interactions_mode == 'protein-duplicate'):
        return corrupt_duplicate_rows(c, batch_dict, dataset_mode, row_index)
    else:
        return corrupt_standard_dataset(c, batch_dict, dataset_mode, row_index)


def duplicate_batch_dict(batch_dict):
    def recursive_clone(obj):
        if isinstance(obj, (int, float)):
            return obj
        elif isinstance(obj, list):
            return [recursive_clone(elem) for elem in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.clone().detach()
        elif obj is None:
            return None
        else:
            raise NotImplementedError

    new_batch_dict = {}
    for key, value in batch_dict.items():
        new_batch_dict[key] = recursive_clone(value)

    return new_batch_dict


def corrupt_duplicate_rows(c, batch_dict, dataset_mode, row_index):
    """
    The aim of this corruption is to show that using the `designated lookup
    row` (located at `row_index`, which is a duplicate of the row at
    `row_index` + N) is necessary to solve the task for duplicated datasets,
    like protein-duplication.

    We wish to remove the ability to perform a successful lookup, and
    accomplish this by "flipping" the label of the duplicated row.
        - We can't simply input just a single row to our model, because
            we'd be changing batch statistics, which could account for
            changes in the prediction.
        - We don't want to corrupt the features as well -- the model should
            still be able to lookup the right row, but then should fail
            because of the label alteration we made.

    We will select a new label to which we flip the label of the designated
    lookup row by selecting uniformly at random from other unmasked rows.
    These unmasked rows are specified by the label_matrix, which is aware
    of stochastic label masking changes.

    Finally, we restrict the label_matrix to assure that we are only
    evaluating a loss on the `row_index`.
    """
    # Avoid overwriting things we will need in corruptions for other rows
    bd = duplicate_batch_dict(batch_dict)

    if bd['label_mask_matrix'] is not None:
        # Triggers for stochastic label masking.
        # Only not None in train mode. In which case we can use it to only
        # reveal those train indices that have been masked.
        label_matrix = 'label'
    else:
        # We are in val/test mode. In which case all val/test labels are masked
        # and need to be revealed at val/test time, to check that model is
        # actually learning interactions!
        label_matrix = dataset_mode

        # (Note that there may be stochastic masking on the train labels still.
        # but we do not reveal those anymore as there is no loss computed on
        # them.)

    if bd[f'{label_matrix}_mask_matrix'] is None:
        raise NotImplementedError

    num_cols = len(bd['data_arrs'])
    num_rows = bd['data_arrs'][0].shape[0] // 2

    # Keep track of target columns -- we will need to zero out the
    # label_matrix, and then set only the row_index in the specified
    # target columns so that we are only evaluating loss on our chosen
    # row index
    target_cols = []

    for col in range(num_cols):
        # get true values wherever the label matrix has masks
        locations = bd[f'{label_matrix}_mask_matrix'][:, col].nonzero(
            as_tuple=True)[0]
        if locations.nelement() == 0:
            continue

        target_cols.append(col)

        # These locations currently give us indexes where the loss should
        # be evaluated. We can determine the locations of the unmasked rows
        # by subtracting the original number of rows.
        locations -= num_rows

        # Next, we remove the provided row_index, as we do not want to flip its
        # label to itself -- this would of course be unsuccessful in corrupting
        # the label!
        locations = locations.tolist()
        locations = list(set(locations) - {row_index})

        # Randomly select one of the locations
        flip_index = np.random.choice(locations)

        # Replace the label of the `designated lookup row` with that of the
        # flip_index row we have just randomly selected
        bd[
            'masked_tensors'][col][row_index] = bd[
            'masked_tensors'][col][flip_index]

    # Only evaluate loss on the row_index in appropriate target columns.

    # Obtain loss index as originally specified row_index + number of rows
    loss_index = row_index + num_rows
    rows_to_zero = list(set(range(int(num_rows * 2))) - {loss_index})
    bd[f'{label_matrix}_mask_matrix'][rows_to_zero, :] = False

    return bd


def corrupt_standard_dataset(c, batch_dict, dataset_mode, row_index):
    """
    The aim of this corruption is to show that using row interactions improves
    performance on a standard dataset, such as protein, higgs, or forest-cover.

    To accomplish this corruption, we independently permute each of the columns
    over all row indices, __excluding__ the specified row index.
    """
    # Avoid overwriting things we will need in corruptions for other rows
    bd = duplicate_batch_dict(batch_dict)

    n_cols = len(bd['data_arrs'])
    n_rows = bd['data_arrs'][0].shape[0]

    # Row indices to shuffle -- exclude the given row_index
    row_indices = list(set(range(n_rows)) - {row_index})

    # Shuffle all rows other than our selected one, row_index
    # Perform an independent permutation for each column so the row info
    # is destroyed (otherwise, our row-equivariant model won't have an
    # issue with permuted rows).
    for col in range(n_cols):
        # Test -- if we ablate shuffle, do not swap around elements
        if not c.debug_corrupt_standard_dataset_ablate_shuffle:
            shuffled_row_indices = np.random.permutation(row_indices)

            # Shuffle masked_tensors, which our model sees at input.
            # Don't need to shuffle data_arrs, because the row at which
            # we evaluate loss will be in the same place.
            bd['masked_tensors'][col][row_indices] = bd[
                'masked_tensors'][col][shuffled_row_indices]

        # We also zero out the
        # {dataset_mode}, augmentation, and label mask matrices at all
        # rows other than row_index
        for matrix in [dataset_mode, 'augmentation', 'label']:
            mask = f'{matrix}_mask_matrix'
            if bd[mask] is not None:
                bd[mask][:, col][row_indices] = False

    return bd


def random_row_perm(N, batch_dict, dataset_mode):
    row_perm = torch.randperm(N)
    num_cols = len(batch_dict['data_arrs'])
    for col in range(num_cols):
        bdc = batch_dict['data_arrs'][col]
        bdc[:] = bdc[row_perm]

        mt = batch_dict['masked_tensors'][col]
        mt[:] = mt[row_perm]

    batch_dict[f'{dataset_mode}_mask_matrix'] = (
        batch_dict[f'{dataset_mode}_mask_matrix'][row_perm])

    return batch_dict


def leakage(c, batch_dict, masked_tensors, label_mask_matrix, dataset_mode):
    if c.data_set != 'breast-cancer':
        raise Exception

    if not (c.model_label_bert_mask_prob[dataset_mode] == 1):
        raise ValueError(
            'Leakage check only supported for deterministic label masking.')

    target_col = masked_tensors[0]
    assert target_col[:, -1].sum() == masked_tensors[0].size(0)
    assert target_col[:, 0].sum() == 0
    assert target_col[:, 1].sum() == 0
    assert label_mask_matrix is None

    n_label_loss_entries = batch_dict[
        f'{dataset_mode}_mask_matrix'].sum()

    print(f'{dataset_mode} mode:')
    print(f'Inputs over {masked_tensors[0].size(0)} rows.')
    print(
        f'Computing label loss at {n_label_loss_entries} entries.')


def protein_duplicate(c, batch_dict, dataset_mode, duplication_mode):
    """Append unmasked copy to the dataset.
    Allows for perfect loss if model exploits row interactions.
    This is version that respects dataset mode.
    Only unveil labels of current dataset mode.
    Currently does not unveil bert masks in copy.
    """
    verbose = True
    if verbose:
        print('Protein-duplicate mode', duplication_mode)


    N_in, D = batch_dict['data_arrs'][0].shape
    num_cols = len(batch_dict['data_arrs'])
    N_out = 2 * N_in
    bd = batch_dict

    if bd['label_mask_matrix'] is not None:
        # Triggers for stochastic label masking.
        # Only not None in train mode. In which case we can use it to only
        # reveal those train indices that have been masked.
        label_matrix = 'label'
    else:
        # We are in val/test mode. In which case all val/test labels are masked
        # and need to be revealed at val/test time, to check that model is
        # actually learning interactions!
        label_matrix = dataset_mode

        # (Note that there may be stochastic masking on the train labels still.
        # but we do not reveal those anymore as there is no loss computed on
        # them.)

    # do the same for each col
    for col in range(num_cols):

        # duplicate real data
        bd['data_arrs'][col] = torch.cat([
            bd['data_arrs'][col],
            bd['data_arrs'][col]], 0)

        # create new copy of data where masks are removed for everything that
        # is currently part of dataset_mode mask matrix
        # (i.e. all the labels)

        # append masked data again
        predict_rows = bd['masked_tensors'][col]
        if ('no-nn' in duplication_mode) and col > 2:
            lookup_rows = torch.ones_like(predict_rows)
            lookup_rows[:, 0] = torch.normal(
                mean=torch.Tensor(N_in*[1.]),
                std=torch.Tensor(N_in*[1.]))

            bd['masked_tensors'][col] = torch.cat([
                lookup_rows, predict_rows], 0)
        else:
            lookup_rows = bd['masked_tensors'][col]
            bd['masked_tensors'][col] = torch.cat([
                lookup_rows, predict_rows], 0)

            # now unveil relevant values
            for matrix in [label_matrix, 'augmentation']:
                if bd[f'{matrix}_mask_matrix'] is None:
                    continue

                # get true values wherever current train/aug matrix has masks
                locations = bd[f'{matrix}_mask_matrix'][:, col].nonzero(
                    as_tuple=True)[0]
                # in these locations replace masked tensors with true data
                dtype = bd['masked_tensors'][col].dtype
                bd['masked_tensors'][col][locations] = (
                    bd['data_arrs'][col][locations].type(dtype))

        if ('target-add' in duplication_mode) and (col in bd['target_cols']):
            bd['masked_tensors'][col][locations] += 1


    # now modify the mask_matrices to fit dimensions of new data
    # (all zeros, don't need to predict on that new data)
    for matrix in [dataset_mode, 'augmentation', 'label']:
        mask = f'{matrix}_mask_matrix'
        if bd[mask] is not None:
            bd[mask] = torch.cat([
                torch.zeros_like(bd[mask]),
                bd[mask]], 0)

    return batch_dict
