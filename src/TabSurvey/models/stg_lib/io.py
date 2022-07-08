import os.path as osp

import numpy as np
import torch
import torch.nn as nn

import logging.config 
from .matching import IENameMatcher
from .utils import as_cpu

import logging
logger = logging.getLogger("my-logger")

__all__ = ['load_state_dict', 'load_weights']


def state_dict(model, include=None, exclude=None, cpu=True):
    if isinstance(model, nn.DataParallel):
        model = model.module

    state_dict = model.state_dict()

    matcher = IENameMatcher(include, exclude)
    with matcher:
        state_dict = {k: v for k, v in state_dict.items() if matcher.match(k)}
    stat = matcher.get_last_stat()
    if len(stat[1]) > 0:
        logger.critical('Weights {}: {}.'.format(stat[0], ', '.join(sorted(list(stat[1])))))

    if cpu:
        state_dict = as_cpu(state_dict)
    return state_dict


def load_state_dict(model, state_dict, include=None, exclude=None):
    if isinstance(model, nn.DataParallel):
        model = model.module

    matcher = IENameMatcher(include, exclude)
    with matcher:
        state_dict = {k: v for k, v in state_dict.items() if matcher.match(k)}
    stat = matcher.get_last_stat()
    if len(stat[1]) > 0:
        logger.critical('Weights {}: {}.'.format(stat[0], ', '.join(sorted(list(stat[1])))))

    # Build the tensors.
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v)

    error_msg = []
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                error_msg.append('While copying the parameter named {}, '
                                 'whose dimensions in the model are {} and '
                                 'whose dimensions in the checkpoint are {}.'
                                 .format(name, own_state[name].size(), param.size()))

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        error_msg.append('Missing keys in state_dict: "{}".'.format(missing))

    unexpected = set(state_dict.keys()) - set(own_state.keys())
    if len(unexpected) > 0:
        error_msg.append('Unexpected key "{}" in state_dict.'.format(unexpected))

    if len(error_msg):
        raise KeyError('\n'.join(error_msg))


def load_weights(model, filename, include=None, exclude=None, return_raw=True):
    if osp.isfile(filename):
        try:
            raw = weights = torch.load(filename)
            # Hack for checkpoint.
            if 'model' in weights and 'optimizer' in weights:
                weights = weights['model']

            try:
                load_state_dict(model, weights, include=include, exclude=exclude)
            except KeyError as e:
                logger.warning('Unexpected or missing weights found:\n' + e.args[0])
            logger.critical('Weights loaded: {}.'.format(filename))
            if return_raw:
                return raw
            return True
        except Exception:
            logger.exception('Error occurred when load weights {}.'.format(filename))
    else:
        logger.warning('No weights file found at specified position: {}.'.format(filename))
    return None