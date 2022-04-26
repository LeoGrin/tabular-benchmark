def load_image_dataloaders(c):
    batch_size = c.exp_batch_size
    from npt.utils.image_loading_utils import get_dataloaders

    if c.data_set in ['cifar10']:
        # For CIFAR, let's just use 10% of the training set for validation.
        # That is, 10% of 50,000 rows = 5,000 rows
        val_perc = 0.10
    else:
        raise NotImplementedError

    _, trainloader, validloader, testloader = get_dataloaders(
        c.data_set, batch=batch_size, dataroot=f'{c.data_path}/{c.data_set}',
        c=c, split=val_perc, split_idx=0)
    data_dict = {
        'trainloader': trainloader,
        'validloader': validloader,
        'testloader': testloader}

    return data_dict
