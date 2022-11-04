import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

from npt.utils.eval_checkpoint_utils import EarlyStopCounter


def plot_grid_query_pix(width, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.set_xticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.set_aspect(1)
    ax.set_yticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(True, alpha=0.5)

    # query pixel
    querry_pix = Rectangle(xy=(-0.5,-0.5),
                          width=1,
                          height=1,
                          edgecolor="black",
                          fc='None',
                          lw=2)

    ax.add_patch(querry_pix);

    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)
    ax.set_aspect("equal")


def plot_attention_layer(attention_probs, axes):
    """Plot the 2D attention probabilities for a particular MAB attention map."""

    contours = np.array([0.9, 0.5])
    linestyles = [":", "-"]
    flat_colors = ["#3498db", "#f1c40f", "#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#34495e", "#1abc9c", "#95a5a6"]

    shape = attention_probs.shape
    num_heads, height, width = shape
    # attention_probs = attention_probs.reshape(width, height, num_heads)

    try:
        ax = axes[0]
    except:
        attention_prob_head = attention_probs[0].detach().cpu().numpy()
        sns.heatmap(attention_prob_head, ax=axes, square=True)
        axes.set_title(f'Head 1')
        return axes

    for head_index in range(num_heads):
        attention_prob_head = attention_probs[head_index].detach().cpu().numpy()
        sns.heatmap(attention_prob_head, ax=axes[head_index], square=True)
        axes[head_index].set_title(f'Head {head_index}')

    return axes


def viz_att_maps(c, dataset):
    early_stop_counter = EarlyStopCounter(
        c=c, data_cache_prefix=dataset.model_cache_path,
        metadata=dataset.metadata,
        device=c.exp_device)

    # Initialize from checkpoint, if available
    num_steps = 0

    checkpoint = early_stop_counter.get_most_recent_checkpoint()
    if checkpoint is not None:
        checkpoint_epoch, (
            model, optimizer, num_steps) = checkpoint
    else:
        raise Exception('Could not find a checkpoint!')

    dataset.set_mode(mode='test', epoch=num_steps)
    batch_dataset = dataset.cv_dataset
    batch_dict = next(batch_dataset)

    from npt.utils import debug

    if c.debug_row_interactions:
        print('Detected debug mode.'
              'Modifying batch input to duplicate rows.')
        batch_dict = debug.modify_data(c, batch_dict, 'test', 0)

    # Run a forward pass
    masked_tensors = batch_dict['masked_tensors']
    masked_tensors = [
        masked_arr.to(device=c.exp_device)
        for masked_arr in masked_tensors]
    model.eval()
    model(masked_tensors)

    # Grab attention maps from SaveAttMaps modules
    # Collect metadata as we go
    layers = []
    att_maps = []

    for name, param in model.named_parameters():
        if 'curr_att_maps' not in name:
            continue

        _, layer, _, _, _ = name.split('.')
        layers.append(int(layer))
        att_maps.append(param)

    n_heads = c.model_num_heads

    from tensorboardX import SummaryWriter

    # create tensorboard writer
    # adapted from https://github.com/epfml/attention-cnn

    if not c.model_checkpoint_key:
        raise NotImplementedError

    save_path = os.path.join(c.viz_att_maps_save_path, c.model_checkpoint_key)
    tensorboard_writer = SummaryWriter(
        logdir=save_path, max_queue=100, flush_secs=10)
    print(f"Tensorboard logs saved in '{save_path}'")

    for i in range(len(att_maps)):
        layer_index = layers[i]
        att_map = att_maps[i]

        # If n_heads != att_map.size(0), we have attention over the
        # columns, which is applied to every
        # one of the batch dimension axes independently
        # e.g. we will have an attention map of shape (n_heads * N, D, D)
        # Just subsample a row for each head
        att_map_first_dim_size = att_map.size(0)
        if n_heads != att_map_first_dim_size:
            print('Subsampling attention over the columns.')
            print(f'Original size: {att_map.size()}')
            n_rows = att_map_first_dim_size // n_heads
            row_subsample_indices = []
            for row_index in range(0, att_map_first_dim_size, n_rows):
                row_subsample_indices.append(row_index)

            att_map = att_map[row_subsample_indices, :, :]
            print(f'Final size: {att_map.size()}')

        fig, axes = plt.subplots(ncols=n_heads, figsize=(15 * n_heads, 15))

        plot_attention_layer(
            att_map, axes=axes)
        if tensorboard_writer:
            tensorboard_writer.add_figure(
                f"attention/layer{layer_index}", fig, global_step=1)
        plt.close(fig)
