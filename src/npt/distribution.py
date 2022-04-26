import numpy as np
import torch
import torch.distributed as dist
import wandb

from npt.column_encoding_dataset import NPTDataset
from npt.train import Trainer
from npt.utils.model_init_utils import (
    init_model_opt_scaler_from_dataset, setup_ddp_model)


def distributed_train_wrapper(gpu, args):
    wandb_args = args['wandb_args']

    if gpu == 0:
        wandb_run = wandb.init(**wandb_args)
        wandb.config.update(args, allow_val_change=True)

    c = args['c']
    rank = c.mp_nr * c.mp_gpus + gpu
    world_size = c.mp_gpus * c.mp_nodes

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)
    torch.manual_seed(c.torch_seed)
    np.random.seed(c.np_seed)

    dataset = args['dataset']
    torch.cuda.set_device(gpu)
    model, optimizer, scaler = init_model_opt_scaler_from_dataset(
        dataset=dataset, c=c, device=gpu)
    model = setup_ddp_model(model=model, c=c, device=gpu)

    distributed_dataset = NPTDataset(dataset)
    dist_args = {
        'world_size': world_size,
        'rank': rank,
        'gpu': gpu}

    trainer = Trainer(
        model=model, optimizer=optimizer, scaler=scaler, c=c,
        cv_index=0, wandb_run=None,
        dataset=dataset,
        torch_dataset=distributed_dataset, distributed_args=dist_args)
    trainer.train_and_eval()

    if gpu == 0:
        wandb_run.finish()