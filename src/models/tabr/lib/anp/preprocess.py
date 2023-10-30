# fmt: off
# isort: off

import torch as t

import numpy as np


def make_context_target(train_size):
    max_num_context = min(500, train_size // 2)
    num_context = np.random.randint(10, max_num_context)
    num_target = np.random.randint(0, max_num_context - num_context)
    num_total_points = num_context + num_target

    ixs = np.random.choice(train_size, size=num_total_points, replace=False)

    return ixs[:num_total_points], ixs


def collate_fn(batch):

    # Puts each data field into a tensor with outer dimension batch size
    assert isinstance(batch[0], tuple)
    trans = torchvision.transforms.ToTensor()
    batch_size = len(batch)

    max_num_context = 784
    num_context = np.random.randint(10, 784)  # extract random number of contexts
    num_target = np.random.randint(0, max_num_context - num_context)
    num_total_points = num_context + num_target  # this num should be # of target points
    #     num_total_points = max_num_context
    context_x, context_y, target_x, target_y = list(), list(), list(), list()

    for d, _ in batch:
        d = trans(d)
        total_idx = np.random.choice(range(784), num_total_points, replace=False)
        total_idx = list(map(lambda x: (x // 28, x % 28), total_idx))
        c_idx = total_idx[:num_context]
        c_x, c_y, total_x, total_y = list(), list(), list(), list()
        for idx in c_idx:
            c_y.append(d[:, idx[0], idx[1]])
            c_x.append((idx[0] / 27.0, idx[1] / 27.0))
        for idx in total_idx:
            total_y.append(d[:, idx[0], idx[1]])
            total_x.append((idx[0] / 27.0, idx[1] / 27.0))
        c_x, c_y, total_x, total_y = list(
            map(lambda x: t.FloatTensor(x), (c_x, c_y, total_x, total_y))
        )
        context_x.append(c_x)
        context_y.append(c_y)
        target_x.append(total_x)
        target_y.append(total_y)

    context_x = t.stack(context_x, dim=0)
    context_y = t.stack(context_y, dim=0).unsqueeze(-1)
    target_x = t.stack(target_x, dim=0)
    target_y = t.stack(target_y, dim=0).unsqueeze(-1)

    return context_x, context_y, target_x, target_y
