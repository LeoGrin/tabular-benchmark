# Feed-forward network

# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import lib
from lib import KWArgs


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: Union[nn.Module, KWArgs]  # Model
    optimizer: Union[torch.optim.Optimizer, KWArgs]  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        backbone: dict,  # lib.deep.ModuleSpec
    ) -> None:
        assert n_num_features or n_bin_features or cat_cardinalities
        if num_embeddings is not None:
            assert n_num_features

        # To support new backbone:
        # - add it to the list below
        # - add it to lib.deep._CUSTOM_MODULES
        # - set the appropriate value for self.flat (see below)
        assert backbone['type'] in ['MLP']
        super().__init__()

        if num_embeddings is None:
            self.m_num = nn.Identity() if n_num_features else None
            d_num = n_num_features
        else:
            self.m_num = lib.make_module(num_embeddings, n_features=n_num_features)
            d_num = n_num_features * num_embeddings['d_embedding']
        self.m_bin = nn.Identity() if n_bin_features else None
        self.m_cat = lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        self.backbone = lib.make_module(
            backbone,
            d_in=d_num + n_bin_features + sum(cat_cardinalities),
            d_out=lib.get_d_out(n_classes),
        )
        # For Transformer-like backbones, which operate over feature embeddings,
        # self.flat must be False.
        # For simple "flat" models (e.g. MLP, ResNet, etc.), self.flat must be True.
        self.flat = True

    def forward(
        self,
        *,
        x_num: Optional[Tensor],
        x_bin: Optional[Tensor],
        x_cat: Optional[Tensor],
    ) -> Tensor:
        x = []
        for module, x_ in [
            (self.m_num, x_num),
            (self.m_bin, x_bin),
            (self.m_cat, x_cat),
        ]:
            if x_ is None:
                assert module is None
            else:
                assert module is not None
                x.append(module(x_))
        del x_  # type: ignore[code]
        if self.flat:
            x = torch.cat([x_.flatten(1, -1) for x_ in x], dim=1)
        else:
            # for Transformer-like backbones
            assert all(x_.ndim == 3 for x_ in x)
            x = torch.cat(x, dim=1)

        x = self.backbone(x)
        return x


def _patch_config(c: lib.JSONDict, logger):
    if isinstance(c['model'], dict):
        if 'backbone' not in c['model']:
            logger.warning('This config format is deprecated')
            assert 'num_embeddings' in c
            c['model'] = {
                'num_embeddings': c.pop('num_embeddings'),
                'backbone': c.pop('model'),
            }


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    _patch_config(config, logger)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    # >>> data
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    Y_train = dataset.Y['train'].to(
        torch.long if dataset.is_multiclass else torch.float
    )

    # >>> model
    if isinstance(C.model, nn.Module):
        model = C.model
    else:
        model = Model(
            n_num_features=dataset.n_num_features,
            n_bin_features=dataset.n_bin_features,
            cat_cardinalities=dataset.cat_cardinalities(),
            n_classes=dataset.n_classes(),
            **C.model,
        )
    report['n_parameters'] = lib.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = None if dataset.is_regression else 'logits'
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # type: ignore[code]

    # >>> training
    optimizer = (
        C.optimizer
        if isinstance(C.optimizer, torch.optim.Optimizer)
        else lib.make_optimizer(model, **C.optimizer)
    )
    loss_fn = lib.get_loss_fn(dataset.task_type)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    def apply_model(part, idx):
        return model(
            x_num=None if dataset.X_num is None else dataset.X_num[part][idx],
            x_bin=None if dataset.X_bin is None else dataset.X_bin[part][idx],
            x_cat=None if dataset.X_cat is None else dataset.X_cat[part][idx],
        ).squeeze(-1)

    @torch.inference_mode()
    def evaluate(parts: list[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in torch.arange(
                                    dataset.size(part), device=device
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        metrics = (
            dataset.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'progress': progress,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    print()
    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(dataset.size('train'), C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda x: loss_fn(apply_model('train', x), Y_train[x]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer()}
        )
        writer.add_scalars('loss', {'train': mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars('score', {part: metrics[part]['score']}, epoch, timer())

        progress.update(metrics['val']['score'])
        if progress.success:
            lib.celebrate()
            report['best_epoch'] = epoch
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()
    report['time'] = str(timer)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
