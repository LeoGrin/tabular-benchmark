import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import six
import collections
import copy
from torch.utils.data import Dataset
from collections import defaultdict
import h5py
from scipy.stats import norm
import os

SKIP_TYPES = six.string_types


class SimpleDataset(Dataset):
    '''
    Assuming X and y are numpy arrays and 
     with X.shape = (n_samples, n_features) 
          y.shape = (n_samples,)
    '''
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i]
        data = np.array(data).astype(np.float32)
        if self.y is not None:
            return dict(input=data, label=self.y[i])
        else:
            return dict(input=data)


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, tensor_names, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param tensor_names: name of tensors (for feed_dict)
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.tensor_names = tensor_names

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = {}
        for k in range(len(self.tensor_names)):
            batch.update({self.tensor_names[k]: self.tensors[k][self.i:self.i+self.batch_size]})
        self.i += self.batch_size
        return batch
        

    def __len__(self):
        return self.n_batches


'''standardize_dataset function is from utils_jared.py'''
def standardize_dataset(dataset, offset, scale):
    norm_ds = copy.deepcopy(dataset)
    norm_ds['x'] = (norm_ds['x'] - offset) / scale
    return norm_ds


'''load_datasets function is from utils_jared.py'''
def load_datasets(dataset_file):
    datasets = defaultdict(dict)
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def load_cox_gaussian_data():
    dataset_file = os.path.join(os.path.dirname(__file__), 
        'datasets/gaussian_survival_data.h5')
    datasets = defaultdict(dict)
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort training data for accurate partial likelihood calculation.
    sort_idx = np.argsort(t)[::-1]
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    #return x, {'e': e, 't': t} this is for parse_data(x, label); see the third line in the parse_data function. 
    #return {'x': x, 'e': e, 't': t}
    return x, e, t

def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 0:
        print('>>> {} >>>'.format(name))
        print(name, s)
        print(v[nps])
        for k, val in extras.items():
            print(k, val, val.sum().item())
        quit()


class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args

def get_batcnnorm(bn, nr_features=None, nr_dims=1):
    if isinstance(bn, nn.Module):
        return bn

    assert 1 <= nr_dims <= 3

    if bn in (True, 'async'):
        clz_name = 'BatchNorm{}d'.format(nr_dims)
        return getattr(nn, clz_name)(nr_features)
    else:
        raise ValueError('Unknown type of batch normalization: {}.'.format(bn))


def get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout

    if dropout is True:
        dropout = 0.5
    if nr_dims == 1:
        return nn.Dropout(dropout, True)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout)


def get_activation(act):
    if isinstance(act, nn.Module):
        return act

    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'identity':
        return Identity()
    elif act_lower == 'relu':
        return nn.ReLU(True)
    elif act_lower == 'selu':
        return nn.SELU(True)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))


def get_optimizer(optimizer, model, *args, **kwargs):
    if isinstance(optimizer, (optim.Optimizer)):
        return optimizer

    if type(optimizer) is str:
        try:
            optimizer = getattr(optim, optimizer)
        except AttributeError:
            raise ValueError('Unknown optimizer type: {}.'.format(optimizer))
    return optimizer(filter(lambda p: p.requires_grad, model.parameters()), *args, **kwargs)
    

def stmap(func, iterable):
    if isinstance(iterable, six.string_types):
        return func(iterable)
    elif isinstance(iterable, (collections.Sequence, collections.UserList)):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (collections.Mapping, collections.UserDict)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)


def _as_tensor(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        return o
    if torch.is_tensor(o):
        return o
    return torch.from_numpy(np.array(o))


def as_tensor(obj):
    return stmap(_as_tensor, obj)


def _as_numpy(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        o = o
    if torch.is_tensor(o):
        return o.cpu().numpy()
    return np.array(o)


def as_numpy(obj):
    return stmap(_as_numpy, obj)


def _as_float(o):
    if isinstance(o, SKIP_TYPES):
        return o
    if torch.is_tensor(o):
        return o.item()
    arr = as_numpy(o)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    return stmap(_as_float, obj)


def _as_cpu(o):
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cpu()
    return o


def as_cpu(obj):
    return stmap(_as_cpu, obj)


## For synthetic dataset creation
import math
from sklearn.datasets import make_moons
from scipy.stats import norm


# Create a simple dataset
def create_twomoon_dataset(n, p):
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=0.1, random_state=None)
    print(y.shape)
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    print(data.shape)
    return data, y


def create_sin_dataset(n,p):
    x1=5*(np.random.uniform(0,1,n)).reshape(-1,1)
    x2=5*(np.random.uniform(0,1,n)).reshape(-1,1)
    y=np.sin(x1)*np.cos(x2)**3
    relevant=np.hstack((x1,x2))
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    return data, y.astype(np.float32)



def create_simple_sin_dataset(n, p):
    '''This dataset was added to provide an example of L1 norm reg failure for presentation.
    '''
    assert p == 2
    x1 = np.random.uniform(-math.pi, math.pi, n).reshape(n ,1)
    x2 = np.random.uniform(-math.pi, math.pi, n).reshape(n, 1)
    y = np.sin(x1)
    data = np.concatenate([x1, x2], axis=1)
    print("data.shape: {}".format(data.shape))
    return data, y
    