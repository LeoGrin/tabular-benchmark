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

import os
import sys
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
from lib.anp.network import LatentModel

new_chunk_size = False


def make_context_target(train_size, batch_size=32):
    '''
    Draw random number of samples (context). And choose a subset of targets.
    Indices are returned.
    '''

    max_num_context = min(500, train_size // 2)
    num_context = np.random.randint(10, max_num_context)
    num_target = np.random.randint(0, max_num_context - num_context)
    num_total_points = num_context + num_target
    ixs = torch.from_numpy(
        np.random.choice(train_size, size=(batch_size, num_total_points))
    )

    return ixs[:, :num_context], ixs


def adjust_learning_rate(optimizer, step_num, lr_base, warmup_step=4000):
    lr = (
        lr_base
        * warmup_step**0.5
        * min(step_num * warmup_step**-1.5, step_num**-0.5)
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: Union[nn.Module, KWArgs]  # Model
    optimizer: Union[torch.optim.Optimizer, KWArgs]  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]
    num_accumulate: int
    eval_batch_size: int
    eval_batch_size_context: int


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_hidden,  # lib.deep.ModuleSpec
        num_self_attention_l,
        num_cross_attention_l,
        num_layers_dec,
    ) -> None:
        assert n_num_features or n_bin_features or cat_cardinalities
        if num_embeddings is not None:
            assert n_num_features
        # assert backbone['type'] in ['MLP', 'ResNet']
        super().__init__()

        if num_embeddings is None:
            self.m_num = nn.Identity() if n_num_features else None
            d_num = n_num_features
        else:
            self.m_num = lib.make_module(num_embeddings, n_features=n_num_features)
            d_num = n_num_features * num_embeddings['d_embedding']
        self.m_bin = nn.Identity() if n_bin_features else None
        self.m_cat = lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None

        self.backbone = LatentModel(
            d_num + n_bin_features + sum(cat_cardinalities) + 1,
            int(d_hidden // 4 * 4),
            num_self_attention_l,
            num_cross_attention_l,
            num_layers_dec,
        ).cuda()  # lib.make_module(

        # self.backbone = LatentModel(d_num + n_bin_features + sum(cat_cardinalities) + 1, int(d_hidden // 4 * 4)).cuda()#lib.make_module(

        #    backbone,
        #    d_in=d_num + n_bin_features + sum(cat_cardinalities),
        #    d_out=lib.get_d_out(n_classes),
        # )
        self.flat = True

    def forward(
        self,
        context_x_num,
        context_x_cat,
        context_x_bin,
        context_y,
        target_x_num,
        target_x_cat,
        target_x_bin,
        target_y=None,
    ) -> Tensor:
        x = []
        for module, x_ in [
            (self.m_num, context_x_num),
            (self.m_bin, context_x_bin),
            (self.m_cat, context_x_cat),
        ]:
            if x_ is None:
                assert module is None
            else:
                assert module is not None
                x.append(module(x_))
        del x_  # type: ignore[code]
        if self.flat:
            x = torch.cat([x_ for x_ in x], dim=-1)
        else:
            # for Transformer-like backbones (currently not supported)
            assert all(x_.ndim == 3 for x_ in x)
            x = torch.cat(x, dim=1)
        x2 = []
        for module, x_ in [
            (self.m_num, target_x_num),
            (self.m_bin, target_x_bin),
            (self.m_cat, target_x_cat),
        ]:
            if x_ is None:
                assert module is None
            else:
                assert module is not None
                x2.append(module(x_))
        del x_  # type: ignore[code]
        if self.flat:
            x2 = torch.cat([x_ for x_ in x2], dim=-1)
        else:
            # for Transformer-like backbones (currently not supported)
            assert all(x_.ndim == 3 for x_ in x2)
            x = torch.cat(x2, dim=1)
        if target_y is None:
            return self.backbone(x, context_y.float(), x2)
        # x.shape, x2.shape, context_y.shape, target_y.shape, flush=True)
        x = self.backbone(x, context_y.float(), x2, target_y.float())

        return x[-1]


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
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
    eval_batch_size = C.eval_batch_size
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]
    data = dataset
    eval_batch_size_context = C.eval_batch_size_context

    def apply_model(part, idx):
        return model(
            x_num=None if dataset.X_num is None else dataset.X_num[part][idx],
            x_bin=None if dataset.X_bin is None else dataset.X_bin[part][idx],
            x_cat=None if dataset.X_cat is None else dataset.X_cat[part][idx],
        ).squeeze(-1)

    @torch.inference_mode()
    def evaluate(parts, eval_batch_size, eval_batch_size_context=1e10):
        predictions = {}

        for part in parts:
            device = data.X_num["train"].device

            context_ixs = torch.randint(
                0,
                dataset.size("train"),
                size=(1, min(eval_batch_size_context, dataset.size("train") // 3)),
            )  # num of context samples for inference

            context_ixs = context_ixs.to(device)

            context_x_num = (
                data.X_num["train"][context_ixs[0]].unsqueeze(0)
                if not (dataset.X_num is None)
                else None
            )
            context_x_bin = (
                data.X_bin["train"][context_ixs[0]].unsqueeze(0)
                if not (dataset.X_bin is None)
                else None
            )
            context_x_cat = (
                data.X_cat["train"][context_ixs[0]].unsqueeze(0)
                if not (dataset.X_cat is None)
                else None
            )
            context_y = data.Y["train"][context_ixs[0]].unsqueeze(0).unsqueeze(-1)

            i = 0
            pred_y = []
            batch = eval_batch_size
            while batch * len(pred_y) < len(data.Y[part]):
                test_x_num = (
                    data.X_num[part][batch * i : batch * (i + 1)].unsqueeze(0)
                    if not (dataset.X_num is None)
                    else None
                )
                target_x_num = (
                    torch.cat([context_x_num, test_x_num], dim=1)
                    if not (dataset.X_num is None)
                    else None
                )
                test_x_bin = (
                    data.X_bin[part][batch * i : batch * (i + 1)].unsqueeze(0)
                    if not (dataset.X_bin is None)
                    else None
                )
                target_x_bin = (
                    torch.cat([context_x_bin, test_x_bin], dim=1)
                    if not (dataset.X_bin is None)
                    else None
                )
                test_x_cat = (
                    data.X_cat[part][batch * i : batch * (i + 1)].unsqueeze(0)
                    if not (dataset.X_cat is None)
                    else None
                )
                target_x_cat = (
                    torch.cat([context_x_cat, test_x_cat], dim=1)
                    if not (dataset.X_cat is None)
                    else None
                )
                out = model(
                    context_x_num,
                    context_x_cat,
                    context_x_bin,
                    context_y,
                    target_x_num,
                    target_x_cat,
                    target_x_bin,
                )[0]
                pred_y.append(out[:, context_x_num.shape[1] :].flatten())
                i += 1

            pred_y = torch.cat(pred_y)[: len(data.Y[part])]
            predictions[part] = pred_y.cpu().numpy()
        return (
            dataset.calculate_metrics(predictions, None),
            predictions,
            eval_batch_size,
        )

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

    timer = lib.run_timer()
    global_step = 0
    while epoch < C.n_epochs:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(dataset.size('train'), C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            global_step += 1
            adjust_learning_rate(optimizer, global_step, C.optimizer['lr'])
            context_ixs, target_ixs = make_context_target(
                dataset.size("train"), batch_size=C.batch_size
            )
            context_ixs = context_ixs.to(device)
            target_ixs = target_ixs.to(device)
            # print(context_ixs.shape, target_ixs.shape, len(context_ixs))
            small_batch_size = C.batch_size // C.num_accumulate
            for i in range(C.num_accumulate):
                target_x_num = (
                    dataset.X_num["train"][
                        target_ixs[small_batch_size * i : small_batch_size * (i + 1)]
                    ]
                    if not (dataset.X_num is None)
                    else None
                )
                target_x_cat = (
                    dataset.X_cat["train"][
                        target_ixs[small_batch_size * i : small_batch_size * (i + 1)]
                    ]
                    if not (dataset.X_cat is None)
                    else None
                )
                target_x_bin = (
                    dataset.X_bin["train"][
                        target_ixs[small_batch_size * i : small_batch_size * (i + 1)]
                    ]
                    if not (dataset.X_bin is None)
                    else None
                )
                target_y = dataset.Y["train"][
                    target_ixs[small_batch_size * i : small_batch_size * (i + 1)]
                ].unsqueeze(-1)
                context_x_num = (
                    target_x_num[:, : (context_ixs).shape[1]]
                    if not (dataset.X_num is None)
                    else None
                )
                context_x_cat = (
                    target_x_cat[:, : (context_ixs).shape[1]]
                    if not (dataset.X_cat is None)
                    else None
                )
                context_x_bin = (
                    target_x_bin[:, : (context_ixs).shape[1]]
                    if not (dataset.X_bin is None)
                    else None
                )

                context_y = target_y[:, : (context_ixs).shape[1]]
                # target_x_num.shape, context_x_num.shape)
                # context_x = context_x.cuda()
                # context_y = context_y.cuda()
                # target_x = target_x.cuda()
                # target_y = target_y.cuda()

                # pass through the latent model
                loss = model(
                    context_x_num,
                    context_x_cat,
                    context_x_bin,
                    context_y,
                    target_x_num,
                    target_x_cat,
                    target_x_bin,
                    target_y,
                )
                # print(loss)
                if len(loss.shape) != 0:
                    loss = loss.mean()
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size, eval_batch_size_context
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
        ['train', 'val', 'test'], eval_batch_size, eval_batch_size_context
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
