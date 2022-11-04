"""From https://github.com/ildoonet/pytorch-randaugment/blob/48b8f509c4bbda93bbe733d98b3fd052b6e4c8ae/RandAugment/data.py#L32"""

import torch
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import SubsetRandomSampler, Sampler
from torchvision.transforms import transforms

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_dataloaders(dataset, batch, dataroot, c, split=0.15, split_idx=0):
    if 'cifar' in dataset:
        if c.model_image_random_crop_and_flip:
            print('Using random crops and flips in data augmentation.')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
            ])
        else:
            print('NOT using random crops and flips in data augmentation.')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=c.np_seed)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    data_loader_nprocs = c.data_loader_nprocs
    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=data_loader_nprocs, pin_memory=True,
        sampler=train_sampler, drop_last=False)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=data_loader_nprocs, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=data_loader_nprocs, pin_memory=True,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
